// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <compute_at_map.h>
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

namespace {

std::vector<IterDomain*> getShardedIterDomains(TensorView* tv) {
  std::vector<IterDomain*> sharded_ids;
  std::copy_if(
      tv->getLeafDomain().begin(),
      tv->getLeafDomain().end(),
      std::back_inserter(sharded_ids),
      [](auto id) { return id->isDeviceDim(); });
  return sharded_ids;
}

} // namespace

std::unordered_set<TensorView*> haveDifferentSharding(
    TensorView* ref,
    std::unordered_set<TensorView*> tvs) {
  std::unordered_set<TensorView*> ret;

  const auto& reference_dom = ref->getLeafDomain();
  FusionGuard fg(ref->fusion());
  auto ca_map = ComputeAtMap(FusionGuard::getCurFusion());
  std::unordered_map<IterDomain*, IterDomain*> concrete_to_reference_map;
  for (auto id : reference_dom) {
    auto ca_id =
        ca_map.getConcreteMappedID(id, IdMappingMode::PERMISSIVE_RESIZE);
    concrete_to_reference_map[ca_id] = id;
  }

  for (auto tv : tvs) {
    if (!(ref->getDeviceMesh().vector() == tv->getDeviceMesh().vector())) {
      ret.insert(tv);
      continue;
    }
    for (auto id : tv->getLeafDomain()) {
      auto ca_id =
          ca_map.getConcreteMappedID(id, IdMappingMode::PERMISSIVE_RESIZE);
      if (concrete_to_reference_map.count(ca_id) > 0) {
        auto ref_id = concrete_to_reference_map.at(ca_id);
        if ((ref_id->isDeviceDim() || id->isDeviceDim()) &&
            ref_id->getParallelType() != id->getParallelType()) {
          ret.insert(tv);
          break;
        }
      }
    }
  }
  return ret;
}

bool isResharding(Expr* expr) {
  std::unordered_set<TensorView*> tvs;
  for (auto tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
    tvs.insert(tv);
  }
  for (auto tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
    tvs.insert(tv);
  }
  if (tvs.empty()) {
    return false;
  }
  auto tv_ref = *tvs.begin();
  tvs.erase(tv_ref);
  return !haveDifferentSharding(tv_ref, tvs).empty();
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

std::set<DeviceIdxType> involvedDevices(Expr* expr) {
  std::set<DeviceIdxType> ret;
  for (auto tvs : {expr->inputs(), expr->outputs()}) {
    for (auto val : tvs) {
      NVF_ERROR(val->isA<TensorView>(), "Val is not a TensorView");
      auto tv = val->as<TensorView>();
      NVF_ERROR(tv->hasDeviceMesh(), "the TensorView has no device mesh");
      auto& mesh = tv->getDeviceMesh().vector();
      std::copy(mesh.begin(), mesh.end(), std::inserter(ret, ret.end()));
    }
  }
  return ret;
}

} // namespace nvfuser
