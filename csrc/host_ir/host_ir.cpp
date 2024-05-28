// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir/container.h>
#include <host_ir/host_ir.h>
#include <ir/builder.h>
#include <ir/cloner.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <ops/all_ops.h>

namespace nvfuser {

namespace hir {

HostUnit::HostUnit(IrBuilderPasskey passkey, std::unique_ptr<Fusion> fusion)
    : Expr(passkey), fusion_(std::make_unique<Fusion>(*fusion)) {
  NVF_ERROR(passkey.ir_container_->isA<hir::HostIrContainer>()); // NOLINT
}

HostUnit::HostUnit(const HostUnit* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      fusion_(std::make_unique<Fusion>(*src->fusion_to_execute())) {}

NVFUSER_DEFINE_CLONE_AND_CREATE(HostUnit)

std::string HostUnit::toString(int indent_size) const {
  std::stringstream ss;
  fusion_->print(ss, false);
  return ss.str();
}

// TODO: implement better ?
std::string HostUnit::toInlineString(int indent_size) const {
  return toString(indent_size);
}

// TODO: implement
bool HostUnit::sameAs(const Statement* other) const {
  return false;
}

PostOnStream::PostOnStream(
    IrBuilderPasskey passkey,
    Expr* host_op,
    std::vector<Val*> inputs,
    std::vector<Val*> outputs)
    : Expr(passkey, std::move(inputs), std::move(outputs), {host_op}) {
  NVF_ERROR(passkey.ir_container_->isA<hir::HostIrContainer>()); // NOLINT
  if (host_op->isA<HostUnit>()) {
    NVF_ERROR(
        this->inputs().size() ==
        host_op->as<HostUnit>()->fusion_to_execute()->inputs().size());
    NVF_ERROR(
        this->outputs().size() ==
        host_op->as<HostUnit>()->fusion_to_execute()->outputs().size());
  } else if (host_op->isA<Communication>()) {
    NVF_ERROR(
        this->inputs().size() == 1,
        "Communication must have exactly one input");
    NVF_ERROR(
        this->outputs().size() == 1,
        "Communication must have exactly one output");
  }
  // TODO: harden the assert checks with smth like
  // for (int i : c10::irange(inputs.size())) {
  //     // NVF_ERROR(inputs.at(i)->sameAs(executable_fusion->inputs().at(i)));
  // }
  // for (int i : c10::irange(outputs.size())) {
  //     //
  //     NVF_ERROR(outputs.at(i)->sameAs(executable_fusion->outputs().at(i)));
  // }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(PostOnStream)

std::string PostOnStream::toString(int indent_size) const {
  int indent_increment = 2;
  std::stringstream ss;
  indent(ss, indent_size) << "PostOnStream the operation :{\n";
  indent(ss, indent_size) << hostOpToPost()->toString();
  indent(ss, indent_size) << "\n}, taking inputs: {";
  for (auto input : inputs()) {
    indent(ss, indent_size + indent_increment)
        << input->toString(indent_size + indent_increment) << "\n";
  }
  indent(ss, indent_size) << "} and outputs: {\n";
  for (auto output : outputs()) {
    indent(ss, indent_size + indent_increment)
        << output->toString(indent_size + indent_increment) << "\n";
  }
  indent(ss, indent_size) << "}" << std::endl;
  return ss.str();
}

// TODO: implement better ?
std::string PostOnStream::toInlineString(int indent_size) const {
  return toString(indent_size);
}

// TODO: implement
bool PostOnStream::sameAs(const Statement* other) const {
  return false;
}

int StreamIr::running_counter_ = 0;

StreamIr::StreamIr(IrBuilderPasskey passkey): Val(passkey, ValType::StreamIr), idx_(running_counter_++) {};

StreamIr::StreamIr(const StreamIr* src, IrCloner* ir_cloner): Val(src, ir_cloner), idx_(src->idx_){};
NVFUSER_DEFINE_CLONE(StreamIr)

std::string StreamIr::toString(int indent_size) const {
    std::stringstream ss;
    indent(ss, indent_size) << "Stream " << idx_;
    return ss.str();
}

std::string StreamIr::toInlineString(int indent_size) const {
    return toString(indent_size);
}

bool StreamIr::sameAs(const Statement* other) const {
    return false;
}

} // namespace hir

} // namespace nvfuser
