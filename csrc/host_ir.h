// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <ir/base_nodes.h>
#include <ir/builder.h>

namespace nvfuser {

namespace hir {

class HostFusion final : public Fusion {
 public:
  HostFusion() = default;
  HostFusion(const HostFusion&) = delete;
  HostFusion& operator=(const HostFusion&) = delete;

  Fusion* gpu_fusion;
};

std::unique_ptr<HostFusion> makeHostFusionFromFusion(Fusion* fusion);

} // namespace hir

} // namespace nvfuser
