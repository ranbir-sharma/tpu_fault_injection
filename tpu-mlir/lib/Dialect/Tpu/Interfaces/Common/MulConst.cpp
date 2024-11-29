//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"


LogicalResult tpu::MulConstOp::init(InferenceParameter &p) { return success(); }

void tpu::MulConstOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::MulConstOp::inference(InferenceParameter &p) {
  auto num_elem = module::getNumElements(getOutput());
  auto out_type = module::getStorageType(getOutput());
  auto asym = module::isAsymmetric();
  double const_v = getConstVal().convertToDouble();
  if (out_type.isFloat8E4M3FN()) {
    const_v = F16(const_v, true);
  }
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    p.outputs[0][i] = p.inputs[0][i] * const_v;
  }
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isFloat8E4M3FN()) {
      F8E4M3(p.outputs[0], p.outputs[0], num_elem, 1.0, true);
    } else if (out_type.isFloat8E5M2()) {
      F8E5M2(p.outputs[0], p.outputs[0], num_elem, 1.0, true);
    }
  } else if (module::isUniformQuantized(getOutput())) {
    if (asym == false) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        // coeff has been merge in multiplier&&rshift
        double sum = applyMultiplierAndRShift(p.inputs[0][i], getMultiplier(),
                                              getRshift());
        if (getDoRelu() && sum < 0) {
          sum = 0;
        }
        p.outputs[0][i] = saturate(sum, out_type);
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        // inputs has been requant
        double sum = p.inputs[0][i];
        if (getDoRelu() && sum < 0) {
          sum = 0;
        }
        p.outputs[0][i] = saturate(sum, out_type);
      }
    }
  }
  return success();
}

LogicalResult tpu::MulConstOp::LocalGenSupport() { return success(); }

void tpu::MulConstOp::assign_fw_param(void *param) {
  IR_PARAM_CONST_BINARY(BINARY_MUL);
}

mlir::Type tpu::MulConstOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  // int32-mul-float is not support. C is float point currently.
  // input/C/output must be all integer or all float point
  auto op = getOperation();
  // if (op && opd_idx == 0) {
  //   auto opd = op->getOperand(0);
  //   auto in_op = opd.getDefiningOp();
  //   if (in_op != nullptr && isa<top::WeightOp, top::NoneOp>(in_op)) {
  //     return do_nothing(mode);
  //   }
  //   auto stype = module::getStorageType(opd);
  //   if (stype.isInteger(32)) {
  //     mode = TypeCastMode::DO_CAST;
  //     return Builder(op).getF32Type();
  //   }
  // }
  // cast input according to output type directly
  return type_verify_case_same(op, opd_idx, mode);
}

ArrayAttr tpu::MulConstOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

// case1: Fuse multiple mulconst ops into one
// only when in_dtype == out_dtype or in_dtype == fp8
class FuseMultiMulConst  : public OpRewriterPatternEx<tpu::MulConstOp> {
  public:
  FuseMultiMulConst(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::MulConstOp>(context,"FuseMultiMulConst") {}

  LogicalResult matchAndRewriteImpl(tpu::MulConstOp op,
                                PatternRewriter &rewriter) const override {

    // starts from the last mulconst op
    if (!op->hasOneUse() ||
        dyn_cast<tpu::MulConstOp>(module::getNextOp(op, 0))) {
      return failure();
    }

    auto input = op.getInput();
    auto in_dtype = BM168x::getDataType(input);
    auto out_dtype = BM168x::getDataType(op.getOutput());
    auto final_const_val = op.getConstVal().convertToDouble();
    auto prev_op = dyn_cast<tpu::MulConstOp>(input.getDefiningOp());
    if (!prev_op) {
      return failure();
    }

    while (in_dtype == out_dtype || in_dtype == DTYPE_F8E4M3 ||
           in_dtype == DTYPE_F8E5M2) {
      final_const_val *= prev_op.getConstVal().convertToDouble();
      input = prev_op.getInput();
      prev_op = dyn_cast<tpu::MulConstOp>(input.getDefiningOp());
      if (!prev_op) {
        break;
      }
      out_dtype = in_dtype;
      in_dtype = BM168x::getDataType(prev_op.getInput());
    }
    op.setConstValAttr(rewriter.getF64FloatAttr(final_const_val));
    op.setOperand(input);

    return success();
  }
  bool shouldPrint(tpu::MulConstOp op) const override { return false;}
};

// case2: Fuse cast to FP8 MulConst
class FuseCastToF8MulConst  : public OpRewriterPatternEx<tpu::MulConstOp> {
  public:
  FuseCastToF8MulConst(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::MulConstOp>(context,"FuseCastToF8MulConst") {}

  LogicalResult matchAndRewriteImpl(tpu::MulConstOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto in_dtype = BM168x::getDataType(input);
    if (!(in_dtype == DTYPE_F8E4M3 || in_dtype == DTYPE_F8E5M2)) {
      return failure();
    }

    if (!op->hasOneUse()) {
      return failure();
    }

    auto castOp = dyn_cast<tpu::CastOp>(module::getNextOp(op, 0));
    if (!castOp) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<tpu::MulConstOp>(
        castOp, castOp.getType(), ValueRange{input}, op->getAttrs());
    return success();
  }
  bool shouldPrint(tpu::MulConstOp op) const override { return false;}
};

void tpu::MulConstOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.insert<FuseMultiMulConst, FuseCastToF8MulConst>(context);
}

bool tpu::MulConstOp::support_multi_core() { return false; }
