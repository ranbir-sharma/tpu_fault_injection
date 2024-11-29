#!/bin/bash
# test case: test addr_mode, base, io_tag, io_alone
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# ===------------------------------------------------------------===
# Qwen
# ===------------------------------------------------------------===
# block
# qwen_0.5b
model_transform.py \
  --model_name qwen_block_0 \
  --model_def ${NNMODELS_PATH}/llm_models/qwen_block_0.onnx \
  --mlir qwen_block_0.mlir

model_runner.py \
  --input ${NNMODELS_PATH}/llm_models/qwen_block_0_input.npz \
  --model qwen_block_0.mlir \
  --output qwen_block_0_top_outputs.npz

model_deploy.py \
  --mlir qwen_block_0.mlir \
  --quantize W4BF16 \
  --q_group_size 64 \
  --chip bm1684x \
  --quant_input \
  --quant_output \
  --debug \
  --model qwen_block_0.bmodel

model_runner.py \
  --input ${NNMODELS_PATH}/llm_models/qwen_block_0_input.npz \
  --model qwen_block_0_bm1684x_w4bf16_tpu.mlir \
  --output qwen_block_0_tpu_outputs.npz

npz_tool.py compare \
  qwen_block_0_top_outputs.npz \
  qwen_block_0_tpu_outputs.npz \
  --tolerance 0.98,0.90 -v

# dynamic
model_deploy.py \
  --mlir qwen_block_0.mlir \
  --quantize W4BF16 \
  --q_group_size 64 \
  --chip bm1684x \
  --dynamic \
  --quant_input \
  --quant_output \
  --model qwen_block_0_dynamic.bmodel

model_runner.py \
  --input ${NNMODELS_PATH}/llm_models/qwen_block_0_input.npz \
  --model qwen_block_0_dynamic.bmodel \
  --output qwen_block_0_model_outputs.npz

npz_tool.py compare \
  qwen_block_0_model_outputs.npz \
  qwen_block_0_tpu_outputs.npz \
  --tolerance 0.98,0.90 -v

# parallel
model_deploy.py \
  --mlir qwen_block_0.mlir \
  --quantize W4BF16 \
  --q_group_size 64 \
  --chip bm1684x \
  --num_device 8 \
  --quant_input \
  --quant_output \
  --model qwen_block_0_dev8.bmodel

# block cache
# qwen_0.5b
model_transform.py \
  --model_name qwen_block_cache_0 \
  --model_def ${NNMODELS_PATH}/llm_models/qwen_block_cache_0.onnx \
  --mlir qwen_block_cache_0.mlir

model_runner.py \
  --input ${NNMODELS_PATH}/llm_models/qwen_block_cache_0_input.npz \
  --model qwen_block_cache_0.mlir \
  --output qwen_block_cache_0_top_outputs.npz

model_deploy.py \
  --mlir qwen_block_cache_0.mlir \
  --quantize W8BF16 \
  --chip bm1684x \
  --test_input ${NNMODELS_PATH}/llm_models/qwen_block_cache_0_input.npz \
  --test_reference qwen_block_cache_0_top_outputs.npz \
  --tolerance 0.98,0.90 \
  --model qwen_block_cache_0_static.bmodel

# dynamic
model_deploy.py \
  --mlir qwen_block_cache_0.mlir \
  --quantize W8BF16 \
  --chip bm1684x \
  --dynamic \
  --test_input ${NNMODELS_PATH}/llm_models/qwen_block_cache_0_input.npz \
  --test_reference qwen_block_cache_0_top_outputs.npz \
  --tolerance 0.98,0.90 \
  --model qwen_block_cache_0_dynamic.bmodel

# parallel
model_deploy.py \
  --mlir qwen_block_cache_0.mlir \
  --quantize W8BF16 \
  --chip bm1684x \
  --num_device 2 \
  --model qwen_block_cache_0_dev2.bmodel

# ===------------------------------------------------------------===
# ChatGLM
# ===------------------------------------------------------------===
# block
# chatglm3_0.5b
model_transform.py \
  --model_name chatglm3_block_0 \
  --model_def ${NNMODELS_PATH}/llm_models/chatglm3_block_0.onnx \
  --mlir chatglm3_block_0.mlir

model_runner.py \
  --input ${NNMODELS_PATH}/llm_models/chatglm3_block_0_input.npz \
  --model chatglm3_block_0.mlir \
  --output chatglm3_block_0_top_outputs.npz

model_deploy.py \
  --mlir chatglm3_block_0.mlir \
  --quantize W4F16 \
  --q_group_size 64 \
  --chip bm1684x \
  --quant_input \
  --quant_output \
  --debug \
  --model chatglm3_block_0.bmodel

model_runner.py \
  --input ${NNMODELS_PATH}/llm_models/chatglm3_block_0_input.npz \
  --model chatglm3_block_0_bm1684x_w4f16_tpu.mlir \
  --output chatglm3_block_0_tpu_outputs.npz

npz_tool.py compare \
  chatglm3_block_0_top_outputs.npz \
  chatglm3_block_0_tpu_outputs.npz \
  --tolerance 0.95,0.77 -v

# parallel
model_deploy.py \
  --mlir chatglm3_block_0.mlir \
  --quantize F16 \
  --chip bm1684x \
  --num_device 2 \
  --quant_input \
  --quant_output \
  --model chatglm3_block_0_dev2.bmodel

# block cache
# chatglm3_0.5b
model_transform.py \
  --model_name chatglm3_block_cache_0 \
  --model_def ${NNMODELS_PATH}/llm_models/chatglm3_block_cache_0.onnx \
  --mlir chatglm3_block_cache_0.mlir

model_runner.py \
  --input ${NNMODELS_PATH}/llm_models/chatglm3_block_cache_0_input.npz \
  --model chatglm3_block_cache_0.mlir \
  --output chatglm3_block_cache_0_top_outputs.npz

model_deploy.py \
  --mlir chatglm3_block_cache_0.mlir \
  --quantize W8F16 \
  --chip bm1684x \
  --test_input ${NNMODELS_PATH}/llm_models/chatglm3_block_cache_0_input.npz \
  --test_reference chatglm3_block_cache_0_top_outputs.npz \
  --tolerance 0.98,0.90 \
  --model chatglm3_block_cache_0_static.bmodel

# parallel
model_deploy.py \
  --mlir chatglm3_block_cache_0.mlir \
  --quantize W8F16 \
  --chip bm1684x \
  --num_device 2 \
  --quant_input \
  --quant_output \
  --model chatglm3_block_cache_0_dev2.bmodel

rm -rf *.npz
rm -rf *.onnx
