#include "config.h"
#include "naive_cpu.h"

#include <alpaka/alpaka.hpp>

#include <cassert>
#include <cstdio>
#include <random>

#include <iostream>
#include <vector>

#include <chrono>
struct CudaLikeSGEMM {
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, auto const in1, auto const in2, auto out, float alpha, float beta) const {
        auto threadIndexMD = acc[alpaka::layer::thread].idx();
        auto blockDimensionMD = acc[alpaka::layer::thread].count();
        auto blockIndexMD = acc[alpaka::layer::block].idx();
        auto gridDimensionMD = acc[alpaka::layer::block].count();
        const int x = blockIndexMD.x() * blockDimensionMD.x() + threadIndexMD.x();
        const int y = blockIndexMD.y() * blockDimensionMD.y() + threadIndexMD.y();
        for (int row = x; row < out.getExtents().x(); row += blockDimensionMD.x()*gridDimensionMD.x()) {
            for (int col = y; col < out.getExtents().y(); col += blockDimensionMD.y()*gridDimensionMD.y()) {
                float tmp = 0.;
                for(int k = 0; k < in1.getExtents().y(); k++) {
                    tmp += in1[Vec2D{k, row}] * in2[Vec2D{col, k}];
                }
                out[Vec2D{col, row}] = alpha * tmp + beta * out[Vec2D{col, row}];
            }
        }
    }
};

struct GMemNonCoalescedKernel {
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, auto const in1, auto const in2, auto out, float alpha, float beta) const {
        for(auto linearIndex :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::linearThreadsInGrid, alpaka::IdxRange{out.getExtents().product()})) {
            int lIndex = linearIndex[0];
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


struct GMemCoalescedKernel {
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, auto const in1, auto const in2, auto out, float alpha, float beta) const {
        for(auto ndIndex :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{out.getExtents()})) {
            auto [col, row] = ndIndex;
            float tmp = 0.0;
            for (auto i = 0; i < in1.getExtents().y(); ++i) {
                tmp += in1[Vec2D{i, row}] * in2[Vec2D{col, i}];
            }
            out[ndIndex] = alpha * tmp + beta * out[ndIndex];
        }
    }
};

struct SMemKernel {
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, auto const in1, auto const in2, auto out, float alpha, float beta) const {
        auto numFramesMD = acc[alpaka::frame::count];
        auto frameExtentMD = acc[alpaka::frame::extent];
        for (auto tileIndexMD : alpaka::onAcc::makeIdxMap(
            acc,
            alpaka::onAcc::worker::blocksInGrid,
            alpaka::IdxRange{numFramesMD})) {
                auto sharedIn1Tile = alpaka::onAcc::declareSharedMdArray<float, alpaka::uniqueId()>(acc, frameExtentMD);
                auto sharedIn2Tile = alpaka::onAcc::declareSharedMdArray<float, alpaka::uniqueId()>(acc, frameExtentMD);
                auto tmp = alpaka::onAcc::declareSharedMdArray<float, alpaka::uniqueId()>(acc, frameExtentMD);
                for (int chunkStride = 0; chunkStride < in1.getExtents().y(); chunkStride+=frameExtentMD.y()) {
                        for (auto tileElemIndexMD : alpaka::onAcc::makeIdxMap(
                            acc,
                            alpaka::onAcc::worker::threadsInBlock,
                            alpaka::IdxRange{frameExtentMD})) {
                                sharedIn1Tile[tileElemIndexMD] = in1[Vec2D{chunkStride+tileElemIndexMD.y(),     frameExtentMD.x()*tileIndexMD.x()+tileElemIndexMD.x()}];
                                sharedIn2Tile[tileElemIndexMD] = in2[Vec2D{frameExtentMD.y()*tileIndexMD.y()+tileElemIndexMD.y(), chunkStride+tileElemIndexMD.x()}];
                                if (chunkStride == 0) tmp[tileElemIndexMD] = 0.;
                            }
                        acc.syncBlockThreads();
                        for (auto tileElemIndexMD : alpaka::onAcc::makeIdxMap(
                            acc,
                            alpaka::onAcc::worker::threadsInBlock,
                            alpaka::IdxRange{frameExtentMD})) {
                                for (int i = 0; i < frameExtentMD.y(); i++) {
                                    tmp[tileElemIndexMD] += sharedIn1Tile[Vec2D{i, tileElemIndexMD.x()}] * sharedIn2Tile[Vec2D{tileElemIndexMD.y(), i}];
                                }
                            }
                        acc.syncBlockThreads();
                    }
                for (auto tileElemIndexMD : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInBlock,
                    alpaka::IdxRange{frameExtentMD})) {
                        out[tileIndexMD*frameExtentMD+tileElemIndexMD] = alpha * tmp[tileElemIndexMD] + beta * out[tileIndexMD*frameExtentMD+tileElemIndexMD];
                    }
	    }
    }
};

struct TestCase {
    Vec2D in1Size;
    Vec2D in2Size;
    Vec2D outSize;
    float alpha, beta;
};

#define FILL(arr, val) \
    for (uint32_t i = 0; i < (arr).getExtents().x(); i++) \
        for (uint32_t j = 0; j < (arr).getExtents().y(); j++) \
            arr[Vec2D{j, i}] = (val);

#define RUN_BENCH(what, iterCount) do { \
    alpaka::onHost::wait(queue); \
    auto start = std::chrono::high_resolution_clock::now(); \
    for (int i = 0; i < iterCount; i++) { \
        queue.enqueue(computeExec, frameSpec, Kernel{}, in1_d.getMdSpan(), in2_d.getMdSpan(), out_d.getMdSpan(), alpha, beta); \
    } \
    alpaka::onHost::wait(queue); \
    auto end = std::chrono::high_resolution_clock::now(); \
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start); \
    what = (uint64_t) duration.count() / iterCount; \
} while (0);

static int API_ID = 0;

template <typename Kernel>
bool benchmark(
    auto algoName,
    auto deviceName,
    alpaka::onHost::concepts::Device auto host,
    alpaka::onHost::concepts::Device auto device,
    auto computeExec,
    TestCase testCase,
    bool onlyOnce)
{
    auto [in1_size, in2_size, out_size, alpha, beta] = testCase;
    assert(in1_size.y() == in2_size.x() && "in1_size.y() != in2_size.x()");
    assert(in1_size.x() == out_size.x() && "in1_size.x() != out_size.x()");
    assert(in2_size.y() == out_size.y() && "in2_size.y() != out_size.y()");
#ifndef ALPAKA_DISABLE_EXEC_CpuOmpBlocksAndThreads
    const char *apiNames[4] = {"CpuSingle", "CpuOmpBlocks", "CpuOmpBlocksAndThreads", "GpuCuda"};
#else
    const char *apiNames[3] = {"CpuSingle", "CpuOmpBlocks", "GpuCuda"};
#endif
    auto apiName = apiNames[API_ID];
    if (deviceName == "Cpu" && out_size.product() > 1024 * 1024) {
        // Skip CPU if the matrices are too big
        return true;
    }
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0.f, 1.f};
    alpaka::onHost::Queue queue = device.makeQueue();

    auto in1_h = alpaka::onHost::alloc<float>(host, in1_size);
    auto in2_h = alpaka::onHost::alloc<float>(host, in2_size);
    auto out_h = alpaka::onHost::alloc<float>(host, out_size);
    auto in1_d = alpaka::onHost::allocMirror(device, in1_h);
    auto in2_d = alpaka::onHost::allocMirror(device, in2_h);
    auto out_d = alpaka::onHost::allocMirror(device, out_h);

    FILL(in1_h, dist(rand));
    FILL(in2_h, dist(rand));
    FILL(out_h, 0.0);

    alpaka::onHost::memcpy(queue, in1_d, in1_h);
    alpaka::onHost::memcpy(queue, in2_d, in2_h);
    alpaka::onHost::memset(queue, out_d, 0x00);

    const int frameExtent1D = 16;
    auto frameExtent = alpaka::CVec<uint32_t, frameExtent1D, frameExtent1D>{};
    int framecountX = std::ceil(out_size.x() / frameExtent.x());
    int framecountY = std::ceil(out_size.y() / frameExtent.y());
    auto frameSpec = alpaka::onHost::FrameSpec{Vec2D{framecountY, framecountX}, frameExtent};

    assert(out_size.x() % frameExtent.x() == 0 && "out_size.x() % frameExtent.x() != 0");
    assert(out_size.y() % frameExtent.y() == 0 && "out_size.y() % frameExtent.y() != 0");
    assert(frameExtent.x() == frameExtent.y() && "frameExtent.x() != frameExtent.y()");

    uint64_t warmup = 0;
    uint64_t warmup_repeats = 5;
    RUN_BENCH(warmup, warmup_repeats)
    uint64_t bench = 0;
    uint64_t bench_repeats = 20;
    RUN_BENCH(bench, bench_repeats)

    bool timeout = bench > 10000000000;
    std::cout <<
        "Alpaka;" << apiName << ";" << algoName << ";" <<
        in1_size << ";" << 
        in2_size << ";" << 
        out_size << ";" <<
        warmup << ";" <<
        bench << ";" <<
        (timeout ? "true" : "false") << ";" <<
    std::endl;
    return timeout;
}

int example(auto const cfg)
{
    auto deviceApi = cfg[alpaka::object::api];
    auto computeExec = cfg[alpaka::object::exec];
    alpaka::onHost::Platform platform = alpaka::onHost::makePlatform(deviceApi);
    std::size_t n = alpaka::onHost::getDeviceCount(platform);
    assert(n != 0);
    alpaka::onHost::Platform host_platform = alpaka::onHost::makePlatform(alpaka::api::cpu);
    alpaka::onHost::Device host = host_platform.makeDevice(0);
    alpaka::onHost::Device device = platform.makeDevice(0);
    auto name = deviceApi.getName();

    bool timeout[4] = {false, false, false, false};
    int step = 16;
    for (int i = 32; i <= 20000; i+=step) {
        if (i % 10 == 0) {
            step *= 2;
        }
        Vec2D size = Vec2D { i, i };
        auto testCase = TestCase{size,size,size,1.0,0.5};
        timeout[0] = benchmark<CudaLikeSGEMM>("CudaLike", name, host, device, computeExec, testCase, timeout[0]);
        timeout[1] = benchmark<GMemNonCoalescedKernel>("Naive", name, host, device, computeExec, testCase, timeout[1]);
        timeout[2] = benchmark<GMemCoalescedKernel>("GMemCoalesced", name, host, device, computeExec, testCase, timeout[2]);
        timeout[3] = benchmark<SMemKernel>("SMemCaching", name, host, device, computeExec, testCase, timeout[3]);
    }
    API_ID++;

    return EXIT_SUCCESS;
}

int main(void) {
    using namespace alpaka;
    // Execute the example once for each enabled API and executor.
    return executeForEach(
        [=](auto const& tag) { return example(tag); },
        onHost::allExecutorsAndApis(onHost::enabledApis));
    return 0;
}