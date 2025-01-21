/* Copyright 2025 Philippe Felix Haupt, Eric-Ramon Kreyer, Andrea Bocci, René Widera
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config.h"
#include "naive_cpu.h"

#include <alpaka/alpaka.hpp>

#include <cassert>
#include <cstdio>
#include <random>

#include <iostream>
#include <vector>

#include <chrono>


struct SMemKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, auto const in1, auto const in2, auto out, float alpha, float beta) const
    {
        auto numFramesMD = acc[alpaka::frame::count];
        auto frameExtentMD = acc[alpaka::frame::extent];

        // Go over each tile
        for (auto tileIndexMD : alpaka::onAcc::makeIdxMap(
            acc,
            alpaka::onAcc::worker::blocksInGrid,
            alpaka::IdxRange{numFramesMD}))
            {
                // this seems like way too much shared memory usage
                auto sharedIn1Tile = alpaka::onAcc::declareSharedMdArray<float, alpaka::uniqueId()>(acc, frameExtentMD);
                auto sharedIn2Tile = alpaka::onAcc::declareSharedMdArray<float, alpaka::uniqueId()>(acc, frameExtentMD);

                auto tmp = alpaka::onAcc::declareSharedMdArray<float, alpaka::uniqueId()>(acc, frameExtentMD);

                // iterate through input buffers with stride of smem size
                // Assumption: frameExtent is quadratic, problem size is dividable by frameExtent
                for (int chunkStride = 0; chunkStride < in1.getExtents().y(); chunkStride+=frameExtentMD.y())
                    {
                        // populate smem
                        for (auto tileElemIndexMD : alpaka::onAcc::makeIdxMap(
                            acc,
                            alpaka::onAcc::worker::threadsInBlock,
                            alpaka::IdxRange{frameExtentMD}))
                            {   // buffer access: in[col, row]
                                // TODO here we could use memory coalescing
                                sharedIn1Tile[tileElemIndexMD] = in1[Vec2D{chunkStride+tileElemIndexMD.y(),     frameExtentMD.x()*tileIndexMD.x()+tileElemIndexMD.x()}];
                                sharedIn2Tile[tileElemIndexMD] = in2[Vec2D{frameExtentMD.y()*tileIndexMD.y()+tileElemIndexMD.y(), chunkStride+tileElemIndexMD.x()}];
                                if (chunkStride == 0) tmp[tileElemIndexMD] = 0.;
                            }

                        /* This call is equal to `alpaka::onAcc::syncBlockThreads(acc)`
                         *
                         * The synchronization is required because we will use for the second loop over frame element indicis a
                         * different traversing schema. Therefore, you should not assume any thread and data element relation.
                         */
                        acc.syncBlockThreads();

                        for (auto tileElemIndexMD : alpaka::onAcc::makeIdxMap(
                            acc,
                            alpaka::onAcc::worker::threadsInBlock,
                            alpaka::IdxRange{frameExtentMD}))
                            {
                                for (int i = 0; i < frameExtentMD.y(); i++)
                                {
                                    tmp[tileElemIndexMD] += sharedIn1Tile[Vec2D{i, tileElemIndexMD.x()}] * sharedIn2Tile[Vec2D{tileElemIndexMD.y(), i}];
                                }
                            }
                        acc.syncBlockThreads();
                    }
                
                for (auto tileElemIndexMD : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInBlock,
                    alpaka::IdxRange{frameExtentMD}))
                    {
                        out[tileIndexMD*frameExtentMD+tileElemIndexMD] = alpha * tmp[tileElemIndexMD] + beta * out[tileIndexMD*frameExtentMD+tileElemIndexMD];
                    }
            }
    }
};

void testGMemNaiveKernel(
    alpaka::onHost::concepts::Device auto host,
    alpaka::onHost::concepts::Device auto device,
    auto computeExec)
{
    // random number generator with a gaussian distribution
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0.f, 1.f};

    // tolerance
    constexpr float epsilon = 0.0001f;

    // 2-dimensional and linearised buffer size
    //constexpr Vec2D in1_size = {1024, 256};
    //constexpr Vec2D in2_size = {256, 1024};
    //constexpr Vec2D out_size = {256, 256};
    constexpr Vec2D in1_size = {4096, 4096};
    constexpr Vec2D in2_size = {4096, 4096};
    constexpr Vec2D out_size = {4096, 4096};
    static_assert(in1_size.y() == in2_size.x());
    static_assert(in1_size.x() == out_size.x());
    static_assert(in2_size.y() == out_size.y());
    float alpha = 1.0;
    float beta = 0.5;

    // allocate input and output host buffers in pinned memory accessible by the Platform devices
    auto in1_h = alpaka::onHost::alloc<float>(host, in1_size);
    auto in2_h = alpaka::onHost::alloc<float>(host, in2_size);
    auto out_h = alpaka::onHost::alloc<float>(host, out_size);

    // fill the input buffers with random data, and the output buffer with zeros
    for (uint32_t i = 0; i < in1_size.x(); ++i)
        for (uint32_t j = 0; j < in1_size.y(); ++j)
            in1_h[Vec2D{j, i}] = dist(rand);
    for (uint32_t i = 0; i < in2_size.x(); ++i)
        for (uint32_t j = 0; j < in2_size.y(); ++j)
            in2_h[Vec2D{j, i}] = dist(rand);
    for (uint32_t i = 0; i < out_size.x(); ++i)
        for (uint32_t j = 0; j < out_size.y(); ++j)
            out_h[Vec2D{j, i}] = 0.;


    // run the test the given device
    alpaka::onHost::Queue queue = device.makeQueue();

    // allocate input and output buffers on the device
    auto in1_d = alpaka::onHost::allocMirror(device, in1_h);
    auto in2_d = alpaka::onHost::allocMirror(device, in2_h);
    auto out_d = alpaka::onHost::allocMirror(device, out_h);

    // copy the input data to the device; the size is known from the buffer objects
    alpaka::onHost::memcpy(queue, in1_d, in1_h);
    alpaka::onHost::memcpy(queue, in2_d, in2_h);

    // fill the output buffer with zeros; the size is known from the buffer objects
    alpaka::onHost::memset(queue, out_d, 0x00);


    const int frameExtent1D = 16;
    auto frameExtent =  alpaka::CVec<uint32_t, frameExtent1D, frameExtent1D>{};
    int framecountX = std::ceil(out_size.x() / frameExtent.x());
    int framecountY = std::ceil(out_size.y() / frameExtent.y());
    auto frameSpec = alpaka::onHost::FrameSpec{Vec2D{framecountY, framecountX}, frameExtent};

    // Assumption for this kernel: Our SMEM is quadratic and cleanly divides sizes of out buffer
    static_assert(out_size.x()%frameExtent.x() == 0);
    static_assert(out_size.y()%frameExtent.y() == 0);
    static_assert(frameExtent.x() == frameExtent.y());

    std::cout << "Testing SMemKernel with scalar indices with a grid of " << frameSpec << "\n";

    alpaka::onHost::wait(queue);
    auto start = std::chrono::high_resolution_clock::now();
    
    queue.enqueue(
        computeExec,
        frameSpec,
        SMemKernel{},
        in1_d.getMdSpan(),
        in2_d.getMdSpan(),
        out_d.getMdSpan(),
        alpha,
        beta);

    alpaka::onHost::wait(queue);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Kernel took " << duration << std::endl;


    // copy the results from the device to the host
    alpaka::onHost::memcpy(queue, out_h, out_d);

    // check the results
    auto cpu_out = alpaka::onHost::allocMirror(host, out_h);
    //alpaka::onHost::memset(queue, cpu_out, 0x00);
    for (int i = 0; i < out_size.x(); i++) {
        for (int j = 0; j < out_size.y(); j++) {
            cpu_out[Vec2D{j, i}] = 0.;
        }
    }

    // wait for all the operations to complete
    alpaka::onHost::wait(queue);

    // Perform a naive CPU matrix multiplication to compare the results
    naive_matrix_mult(in1_h, in2_h, cpu_out, alpha, beta);

    for (uint32_t i = 0; i < out_size.product(); ++i)
    {
        auto lIdx = alpaka::mapToND(out_size, i);
        if(!(std::abs(out_h[lIdx] - cpu_out[lIdx]) < epsilon))
            std::cout << "MISMATCH at " << lIdx << " kernel=" << out_h[lIdx] << " cpu=" << cpu_out[lIdx] << std::endl;
        assert(std::abs(out_h[lIdx] - cpu_out[lIdx]) < epsilon);
    }

    std::cout << "success\n";
}

int example(auto const cfg)
{
    auto deviceApi = cfg[alpaka::object::api];
    auto computeExec = cfg[alpaka::object::exec];

    // initialise the accelerator platform
    alpaka::onHost::Platform platform = alpaka::onHost::makePlatform(deviceApi);

    // require at least one device
    std::size_t n = alpaka::onHost::getDeviceCount(platform);

    if(n == 0)
    {
        return EXIT_FAILURE;
    }

    // use the single host device
    alpaka::onHost::Platform host_platform = alpaka::onHost::makePlatform(alpaka::api::cpu);
    alpaka::onHost::Device host = host_platform.makeDevice(0);
    std::cout << "Host:   " << alpaka::onHost::getName(host) << "\n\n";

    // use the first device
    alpaka::onHost::Device device = platform.makeDevice(0);
    std::cout << "Device: " << alpaka::onHost::getName(device) << "\n\n";

    testGMemNaiveKernel(host, device, computeExec);

    return EXIT_SUCCESS;
}

auto main() -> int
{
    using namespace alpaka;
    // Execute the example once for each enabled API and executor.
    return executeForEach(
        [=](auto const& tag) { return example(tag); },
        onHost::allExecutorsAndApis(onHost::enabledApis));
}
