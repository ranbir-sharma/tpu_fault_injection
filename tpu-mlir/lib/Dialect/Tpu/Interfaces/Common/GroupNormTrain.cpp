//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"

LogicalResult tpu::GroupNormTrainOp::init(InferenceParameter &p) {

  return success();
}

void tpu::GroupNormTrainOp::deinit(InferenceParameter &p) {

}

LogicalResult tpu::GroupNormTrainOp::inference(InferenceParameter &p) {

  return success();
}

uint32_t tpu::GroupNormTrainOp::dyn_codegen_global_bm1684(void* ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}

int64_t tpu::GroupNormTrainOp::dyn_codegen_global_bm1684x(void* ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}

int64_t tpu::GroupNormTrainOp::get_fw_type_bm1684() {
  return -1;
}

int64_t tpu::GroupNormTrainOp::get_fw_type_bm1684x() {
  return -1;
}

bool tpu::GroupNormTrainOp::support_multi_core() { return false; }
