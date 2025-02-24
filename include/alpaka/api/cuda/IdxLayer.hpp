/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/cuda/IdxLayer.hpp"
#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_CUDA

namespace alpaka::onAcc
{
    namespace unifiedCudaHip
    {
        template<typename T_IdxType, uint32_t T_dim>
        struct BlockLayer
        {
            constexpr auto idx() const
            {
                return Vec<T_IdxType, 3u>{::blockIdx.z, ::blockIdx.y, ::blockIdx.x}.template rshrink<T_dim>();
            }

            constexpr auto count() const
            {
                return Vec<T_IdxType, 3u>{::gridDim.z, ::gridDim.y, ::gridDim.x}.template rshrink<T_dim>();
            }
        };

        template<typename T_IdxType, uint32_t T_dim>
        struct ThreadLayer
        {
            constexpr auto idx() const
            {
                return Vec<T_IdxType, 3u>{::threadIdx.z, ::threadIdx.y, ::threadIdx.x}.template rshrink<T_dim>();
            }

            constexpr auto count() const
            {
                return Vec<T_IdxType, 3u>{::blockDim.z, ::blockDim.y, ::blockDim.x}.template rshrink<T_dim>();
            }
        };
    } // namespace unifiedCudaHip
} // namespace alpaka::onAcc

#endif
