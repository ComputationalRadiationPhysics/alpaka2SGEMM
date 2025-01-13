/* Copyright 2025 Philippe Felix Haupt, Eric-Ramon Kreyer, René Widera
 * SPDX-License-Identifier: Apache-2.0
 */

#include <alpaka/alpaka.hpp>

void naive_matrix_mult(
    auto A,
    auto B,
    auto C,
    const float alpha, const float beta)
{
    auto A_dim = A.getExtents();
    auto B_dim = B.getExtents();
    auto C_dim = C.getExtents();
    for (uint32_t i = 0; i < C_dim.x(); ++i)
    {
        for (uint32_t j = 0; j < C_dim.y(); ++j)
        {
            float sum = 0.0;
            for (uint32_t k = 0; k < A_dim.y(); ++k)
            {
                sum += A[Vec2D{k, i}] * B[Vec2D{j, k}];
            }
            C[Vec2D{j, i}] = alpha * sum + beta * C[Vec2D{j, i}];
        }
    }
}