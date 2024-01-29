// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/interface_nodes.h>
#include <multidevice/multidevice.h>

namespace nvfuser {

// Returns whether a TensorView has its first non-reduction axis parallelized
// on Didx
// Checks that the other non-reduction axis are not parallelized on Didx
bool isSharded(TensorView*);

// Returns the axis that is parallelized with type
int dimWithParallelType(TensorView*, ParallelType, bool withReductions=false);

// Returns the subset of tvs which elements have the same multi-device sharding
// as ref
std::unordered_set<TensorView*> haveDifferentSharding(
    TensorView* ref,
    std::unordered_set<TensorView*> tvs);

// Returns whether an Expr embbeds multi-device resharding
bool isResharding(Expr* expr);

// Returns the devices involved in an expr
std::set<DeviceIdxType> involvedDevices(Expr* expr);

// returns the number of device indices present accross all
// device meshes in the Fusion
int64_t requestedNumberOfDevices(Fusion*);

// returns whether a device's sharded tensor is contiguous
// with respect to its unsharded tensor
bool isContiguousShard(TensorView*);

void unshard(Fusion*);
void unshard(TensorView*);
} // namespace nvfuser
