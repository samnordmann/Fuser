// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/internal_base_nodes.h>
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

} // namespace nvfuser
