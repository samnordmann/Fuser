// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <ir/utils.h>
#include <multidevice/allocator.h>
#include <multidevice/executor.h>
#include <multidevice/lower_communication.h>
#include <multidevice/pipeline.h>
#include <multidevice/utils.h>

namespace nvfuser {

bool PipelineExecutor::shouldRun(PipelineStage* stage) {
  if (should_run_.find(stage) == should_run_.end()) {
    should_run_.emplace(
        stage, stage->descriptor()->mesh.has(runtime_.comm_.deviceId()));
  }
  return should_run_[stage];
}

void PipelineExecutor::handle(PipelineStage* stage) {
  if (!shouldRun(stage)) {
    return;
  }
  // get the IValues corresponding to the stage's input
  std::vector<c10::IValue> stage_input_IValues;
  for (auto& input : stage->inputs()) {
    auto input_val = input->as<PipelineVal>()->getOriginalVal();
    NVF_ERROR(val_to_IValue_.find(input_val) != val_to_IValue_.end(), "Device ", runtime_.comm_.deviceId(), " has no buffer associated with Val ", input_val, " for handling stage ", stage);
    NVF_ERROR(val_to_IValue_.at(input_val).isTensor());
    stage_input_IValues.push_back(val_to_IValue_.at(input_val));
  }

  std::vector<at::Tensor> outputs;

  // Compile the stage and either execute it or allocate output buffers
  // if the stage is configured to be autoscheduled, use FusionExecutorCache,
  // otherwise use FusionExecutor
  if (stage->descriptor()->auto_schedule) {
    // Check if the executor has been cached. If not, create and cache it
    if (fec_.find(stage) == fec_.end()) {
      fec_.emplace(
          stage,
          std::make_unique<FusionExecutorCache>(
              runtime_.pipeline_->stageToFusion(stage)));
    }
    // Run the stage to get concrete outputs or placeholders
    outputs = fec_[stage]->runFusionWithInputs(stage_input_IValues);

  } else {
    // Check if the executor has been cached. If not, create and cache it
    if (fe_.find(stage) == fe_.end()) {
      fe_.emplace(stage, std::make_unique<FusionExecutor>());
      fe_[stage]->compileFusion(
          runtime_.pipeline_->stageToFusion(stage).get(), stage_input_IValues);
    } 
    // Run the stage to get concrete outputs or placeholders
    // TODO: deal with aliases I/O. For example if the stage is empty, i.e., Inputs=Outputs, we need to handle them anyway
    outputs = fe_[stage]->runFusion(stage_input_IValues);
  }

  // Store the outputs or placeholders in the context
  for (auto output_idx : c10::irange(outputs.size())) {
    val_to_IValue_[stage->outputs().at(output_idx)->as<PipelineVal>()->getOriginalVal()] = outputs.at(output_idx);
  }
}

void PipelineExecutor::handle(PipelineCommunication* c) {
  auto input_val = c->in()->as<PipelineVal>()->getOriginalVal();
  auto output_val = c->out()->as<PipelineVal>()->getOriginalVal();
  at::Tensor input_tensor, output_tensor;
  if (val_to_IValue_.find(input_val) != val_to_IValue_.end()) {
    input_tensor = val_to_IValue_.at(input_val).toTensor();
  }
  if (val_to_IValue_.find(output_val) != val_to_IValue_.end()) {
    output_tensor = val_to_IValue_.at(output_val).toTensor();
  }

  int input_sharded_dim = dimWithParallelType(static_cast<TensorView*>(input_val), ParallelType::DIDx);
  int output_sharded_dim = dimWithParallelType(static_cast<TensorView*>(output_val), ParallelType::DIDx);
  bool relayout = (input_sharded_dim > 0 && output_sharded_dim == -1);
  auto output_ = output_tensor;
  std::vector<int64_t> permute_order;
  if (relayout) {
    // input tensor was sharded like [a b .. DIDx ... c d] where DIDx is the axis parallelized on DIDx (input_sharded_dim)
    // the output tensor elements will be ordered as [DIDx a b c d] with device dimensions pushed to the outer most axis
    // TODO: for tensor sharded over multiple dimensions it will be [DIDz DIDy DIDx ...]
    // TODO: should probably do this analysis on the tensorview's shape
    auto shape = output_tensor.sizes();
    std::vector<int64_t> written_shape;
    written_shape.push_back(shape[input_sharded_dim]);
    int permute_offset = 1;
    for (int i = 0; i < output_tensor.dim(); i++) {
      if (i == input_sharded_dim) {
        permute_order.push_back(0);
        permute_offset--;
      } else {
        written_shape.push_back(shape[i]);
        permute_order.push_back(i + permute_offset);
      }
    }
    output_ = at::randn(written_shape, output_tensor.options());
  }

  // Lower the Communication into a vector of Communications
  if (communications_.find(c) == communications_.end()) { // check if cached
    communications_.emplace(
        c,
        lowerCommunication(
            runtime_.comm_.deviceId(), c, input_tensor, output_));
  }
  auto& communications = communications_[c];

  // post and wait communications
  for (auto& communication : communications) {
    auto work = communication->post(runtime_.comm_);
    if (work) work->wait();
  }

  if (relayout) {
    // Permute the axis into the correct order and copy into output_tensor.
    auto goal = output_.permute(permute_order);
    output_tensor.copy_(goal);
  }
}

std::vector<at::Tensor> PipelineExecutor::runWithInput(
    const std::vector<c10::IValue>& inputs) {
  // Make sure inputs align at global boundary.
  NVF_ERROR(
      inputs.size() == runtime_.pipeline_->inputs().size(),
      "Wrong number of inputs");

  val_to_IValue_ = allocatePipelineIntermediateBuffers(runtime_.pipeline_, runtime_.comm().deviceId(), inputs);

  // process input values:
  for (auto input_idx : c10::irange(inputs.size())) {
    val_to_IValue_[runtime_.pipeline_->inputs().at(input_idx)->as<PipelineVal>()->getOriginalVal()] =
        inputs.at(input_idx);
  }

  // Run through the stages to launch kernel
  traverseTo(runtime_.pipeline_->outputs());

  // Collect global outputs from context
  std::vector<at::Tensor> outputs;
  for (auto output_val : runtime_.pipeline_->outputs()) {
    outputs.push_back(val_to_IValue_.at(output_val->as<PipelineVal>()->getOriginalVal()).toTensor());
  }

  return outputs;
}

} // namespace nvfuser

#endif
