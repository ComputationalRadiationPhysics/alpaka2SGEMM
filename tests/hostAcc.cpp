/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <functional>
#include <iostream>
#include <thread>

using namespace alpaka;

TEST_CASE("mapping::cpuBlockSerialThreadOne", "")
{
    auto acc = makeAcc(mapping::cpuBlockSerialThreadOne, ThreadBlocking{Vec{4}, Vec{1}});
    acc(KernelBundle{[](auto const& acc)
                     {
                         std::cout << "blockIdx = " << acc[layer::block].idx()
                                   << " threadIdx = " << acc[layer::thread].idx() << std::endl;
                     }});
}
