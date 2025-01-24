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

struct GMemCoalescedKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, auto const in1, auto const in2, auto out, float alpha, float beta) const
    {
        for(auto linearIndex :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::linearThreadsInGrid, alpaka::IdxRange{out.getExtents().product()}))
        {
            int lIndex = linearIndex[0];

            // This index calculation emulates what a naive CUDA implementation might do
            int row = lIndex / out.getExtents().y();
            int col = lIndex % out.getExtents().y();

            float tmp = 0.0;
            for (auto i = 0; i < in1.getExtents().y(); ++i) {
                tmp += in1[Vec2D{i, row}] * in2[Vec2D{col, i}];
            }
            out[Vec2D{col, row}] = alpha * tmp + beta * out[Vec2D{col, row}];
            
        }
    }
};

void testGMemCoalescedKernel(
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
    // constexpr Vec2D in1_size = {1024, 256};
    // constexpr Vec2D in2_size = {256, 1024};
    // constexpr Vec2D out_size = {256, 256};
    // constexpr Vec2D in1_size = {10, 8};
    // constexpr Vec2D in2_size = {4, 10};
    // constexpr Vec2D out_size = {4, 8};
    // constexpr Vec2D in1_size = {10, 4};
    // constexpr Vec2D in2_size = {8, 10};
    // constexpr Vec2D out_size = {8, 4};
    constexpr Vec2D in1_size = {2048, 2048};
    constexpr Vec2D in2_size = {2048, 2048};
    constexpr Vec2D out_size = {2048, 2048};
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
    alpaka::onHost::Queue queue2 = host.makeQueue();

    // allocate input and output buffers on the device
    auto in1_d = alpaka::onHost::allocMirror(device, in1_h);
    auto in2_d = alpaka::onHost::allocMirror(device, in2_h);
    auto out_d = alpaka::onHost::allocMirror(device, out_h);

    // copy the input data to the device; the size is known from the buffer objects
    alpaka::onHost::memcpy(queue, in1_d, in1_h);
    alpaka::onHost::memcpy(queue, in2_d, in2_h);

    // fill the output buffer with zeros; the size is known from the buffer objects
    alpaka::onHost::memset(queue, out_d, 0x00);

    auto frameSpec = alpaka::onHost::FrameSpec{Vec2D{8, 8}, Vec2D{32, 32}};

    alpaka::onHost::wait(queue);
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Testing GMemCoalescedKernel with scalar indices with a grid of " << frameSpec << "\n";
    queue.enqueue(
        computeExec,
        frameSpec,
        GMemCoalescedKernel{},
        in1_d.getMdSpan(),
        in2_d.getMdSpan(),
        out_d.getMdSpan(),
        alpha,
        beta);

    alpaka::onHost::wait(queue);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Kernel took " << duration<< std::endl;

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

    testGMemCoalescedKernel(host, device, computeExec);

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
