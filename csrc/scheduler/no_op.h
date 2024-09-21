// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <scheduler/heuristic.h>
#include <scheduler/registry.h>

namespace nvfuser {

class Fusion;
class SchedulerRuntimeInfo;
class HeuristicSummary;

//! NoOp scheduler represents the case where scheduler will
//!  not do any scheduling operations and forward the un-scheduled
//!  fusion directly to code generation and kernel compilation.
//!
//! Typical use case of this scheduler is to handle edge cases
//!  such as where all tensors are size-1 or size-0.

class NoOpScheduler : public SchedulerEntry {
 public:
  //! Check if the no-op heuristics apply in given fusion
  bool canScheduleCompileTime(Fusion* fusion) override;

  bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) override;

  constexpr static SchedulerType schedulerType() {
    return SchedulerType::NoOp;
  }

  void schedule(Fusion* fusion, const HeuristicParams* params) override;

  std::unique_ptr<HeuristicParams> computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache) override;
};

//! Provides a dummy heuristic type to ensure
//!  unified interface on NoOp scheduler.
class NoOpHeuristic : public HeuristicParams {
 public:
  using HeuristicParams::HeuristicParams;
  NoOpHeuristic() : HeuristicParams(SchedulerType::NoOp) {};

  size_t hash() const override {
    return 0;
  }
  std::unique_ptr<HeuristicParams> clone() const override {
    return std::make_unique<NoOpHeuristic>(*this);
  }

  bool sameAs(const HeuristicParams* other) const override {
    auto other_casted = dynamic_cast<const NoOpHeuristic*>(other);
    return other_casted != nullptr && other_casted->cparams == cparams;
  };
};

} // namespace nvfuser
