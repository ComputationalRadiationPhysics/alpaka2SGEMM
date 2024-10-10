/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_CUDA

#    include "alpaka/Vec.hpp"
#    include "alpaka/acc/Layer.hpp"

#    include <stdexcept>
#    include <tuple>

namespace alpaka
{
    template<typename T_IdxType, uint32_t T_dim>
    struct CudaBlock
    {
        constexpr auto idx() const
        {
            return Vec<T_IdxType, 3u>{::blockIdx.z, ::blockIdx.y, ::blockIdx.x}.template shrink<T_dim>();
        }

        constexpr auto count() const
        {
            return Vec<T_IdxType, 3u>{::gridDim.z, ::gridDim.y, ::gridDim.x}.template shrink<T_dim>();
        }

        template<uint32_t T_idx>
        ALPAKA_FN_ACC void call(auto& acc, auto const& kernelBundle)
        {
            acc.template getLayer<T_idx>().template call<T_idx + 1>(acc, kernelBundle);
        }
    };

    template<typename T_IdxType, uint32_t T_dim>
    struct CudaThread
    {
        constexpr auto idx() const
        {
            return Vec<T_IdxType, 3u>{::threadIdx.z, ::threadIdx.y, ::threadIdx.x}.template shrink<T_dim>();
        }

        constexpr auto count() const
        {
            return Vec<T_IdxType, 3u>{::blockDim.z, ::blockDim.y, ::blockDim.x}.template shrink<T_dim>();
        }

        template<uint32_t T_idx>
        ALPAKA_FN_ACC void call(auto& acc, auto const& kernelBundle)
        {
            acc.template getLayer<T_idx>().template call<T_idx + 1>(acc, kernelBundle);
        }
    };
} // namespace alpaka

#endif
