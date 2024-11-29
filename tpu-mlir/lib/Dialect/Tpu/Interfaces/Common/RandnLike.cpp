//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::RandnLikeOp::init(InferenceParameter &p) { return success(); }
void tpu::RandnLikeOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::RandnLikeOp::inference(InferenceParameter &p) {
  if (p.inputs[0] != p.outputs[0]) {
    auto num_elem = module::getNumElements(getOutput());
    memcpy(p.outputs[0], p.inputs[0], num_elem * sizeof(float));
  }
  return success();
}

bool tpu::RandnLikeOp::support_multi_core() { return false; }
