// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/internal_base_nodes.h>
#include <ir/utils.h>
#include <multidevice/utils.h>

#include <c10/util/irange.h>

namespace nvfuser {

bool isSharded(TensorView* tv) {
  NVF_ERROR(tv->getMaybeRFactorDomain() == tv->getLeafDomain());
  bool is_sharded = false;
  for (IterDomain* id : TensorDomain::noReductions(tv->getLeafDomain())) {
    auto sharded_on_didx = id->getParallelType() == ParallelType::DIDx;
    NVF_ERROR(
      !(is_sharded && sharded_on_didx),
      "Cannot shard multiple axis on the same device dimension");
    is_sharded = is_sharded || sharded_on_didx;
  }
  return is_sharded;
}

int dimWithParallelType(TensorView* tv, ParallelType pt) {
  for (size_t i = 0; i < tv->nDims(); ++i) {
    if (tv->axis(i)->getParallelType() == pt) {
      return i;
    }
  }
  return -1;
}

int64_t requestedNumberOfDevices(Fusion* fusion) {
  std::set<DeviceIdxType> device_indices;
  for (auto tv : ir_utils::filterByType<TensorView>(fusion->vals())) {
    if (tv->hasDeviceMesh()) {
      std::copy(
          tv->getDeviceMesh().vector().begin(),
          tv->getDeviceMesh().vector().end(),
          std::inserter(device_indices, device_indices.begin()));
    }
  }
  return static_cast<int64_t>(device_indices.size());
}

bool isContiguousShard(TensorView* tv) {
  // A shard is contiguous wrt the unsharded tensor if only the 
  // outermost axis are device parallel. 
  auto ids = TensorDomain::noReductions(tv->getLeafDomain());
  bool outermost_sharded = ids[0]->isDeviceDim();
  for (IterDomain* id : ids) {
    auto id_sharded = id->isDeviceDim();
    if (!outermost_sharded && id_sharded) {
      return false;
    }
    outermost_sharded &= id_sharded;
  }
  return true;
}

void unshard(TensorView* tv) {
  for (IterDomain* id : tv->getLeafDomain()) {
    if (id->isDeviceDim()) {
      id->parallelize(ParallelType::Serial);
    }
  }
}

void unshard(Fusion* fusion) {
  for (auto tv : ir_utils::filterByType<TensorView>(fusion->vals())) {
    unshard(tv);
  }
}

} // namespace nvfuser
