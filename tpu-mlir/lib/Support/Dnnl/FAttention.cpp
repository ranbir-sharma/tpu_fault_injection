//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/FAttention.h"
#include "tpu_mlir/Support/Dnnl/Binary.h"
#include "tpu_mlir/Support/Dnnl/MatMul.h"
#include "tpu_mlir/Support/Dnnl/Softmax.h"
#include "tpu_mlir/Support/Float16.h"

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

namespace tpu_mlir {

FAttention::FAttention() {}

void FAttention::setup(float *queries, float *keys, float *values, float *mask,
                       float *output, int64_t batch, int64_t M_q, int64_t M_k,
                       int64_t head, int64_t d, float scale, int dtype) {
                        #if 0
  // int64_t d = K / head;
  scale_ = scale;
  M_k_ = M_k;
  M_q_ = M_q;
  batch_ = batch;
  head_ = head;
  dtype_ = dtype;
  p_mask = mask;
  // matmul0
  num_elem = batch * head * M_q * M_k;
  num_elem_out = batch * M_q * head * d;
  matmul0 = new MatMul();
  data_0 = std::make_shared<std::vector<float>>(num_elem);
  p_mat0 = data_0->data();
  ((MatMul *)matmul0)
      ->setup(queries, keys, nullptr, p_mat0, batch, head, M_q, d, M_k, 0, -1,
              0, 0, 1, 1, 0, 1);
  std::vector<int64_t> lshape = {batch, head, M_q, M_k};
  // binary
  if (mask != nullptr) {
    data_binary = std::make_shared<std::vector<float>>(num_elem);
    p_binary = data_binary->data();
    binary = new Binary();
    // std::vector<int64_t> lshape = {batch, M_q, M_k};
    std::vector<int64_t> rshape = {batch, head, M_q, M_k};
    (*(Binary *)binary)
        .hs(p_mat0, mask, lshape, rshape)
        .dst(p_binary, lshape)
        .algorithem(algorithm::binary_add)
        .setup();
  } else {
    p_binary = p_mat0;
  }
  // // softmax
  softmax = new Softmax();
  softmax_attr_t attr;
  attr.src_shape = lshape;
  attr.dst_shape = lshape;
  attr.axis = 3;
  attr.log = 0;
  data_softmax = std::make_shared<std::vector<float>>(num_elem);
  p_softmax = data_softmax->data();
  ((Softmax *)softmax)->setup(p_binary, p_softmax, attr);
  // matmul1
  matmul1 = new MatMul();
  ((MatMul *)matmul1)
      ->setup(p_softmax, values, nullptr, output, batch, head, M_q, M_k, d, 0,
              -1, 0, 0, 1, 0, 1, 1);
  p_output = output;
  #endif
}

// type = {0:fp32, 1:fp16, 2:bf16, 3:int8}
static void type_cast(float *data, int64_t num, int type) {
  if (type == 1) {
    F16(data, data, num);
  } else if (type == 2) {
    BF16(data, data, num);
  }
  return;
}

void FAttention::run() {
  assert(0);
  #if 0
  int mode = dtype_;
  if (dtype_ < 3) {
    ((MatMul *)matmul0)->run();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      p_mat0[i] *= scale_;
    }
    if (binary != nullptr) {
      ((Binary *)binary)->run();
    }
    ((Softmax *)softmax)->run();
    type_cast(p_softmax, data_softmax->size(), mode);
    ((MatMul *)matmul1)->run();
    type_cast(p_output, num_elem_out, mode);
  } else {
    llvm_unreachable("not supported\n");
  }
  #endif
}

void FAttention::deinit() {
  // delete ((MatMul *)matmul0);
  // if (binary != nullptr)
  //   delete ((Binary *)binary);
  // delete ((Softmax *)softmax);
  // delete ((MatMul *)matmul1);
}

} // namespace tpu_mlir
