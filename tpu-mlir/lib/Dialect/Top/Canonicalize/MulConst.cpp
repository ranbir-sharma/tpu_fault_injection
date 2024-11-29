//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//


#include "tpu_mlir/Support/OpRewriterPatternEx.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;

// merge continuous mulconst, maybe final const_val is 1 and match
// RemoveMulConst below
// TODO : merge all mulconst

struct MergeContinuousMulConst : public OpRewriterPatternEx<MulConstOp> {
  using OpRewriterPatternEx<MulConstOp>::OpRewriterPatternEx;

  MergeContinuousMulConst(MLIRContext *context)
      : OpRewriterPatternEx<MulConstOp>(context, "MergeContinuousMulConst") {}

protected:
  LogicalResult matchAndRewriteImpl(MulConstOp op,
                                    PatternRewriter &rewriter) const override {
    // placeholder
    double const_val = op.getConstVal().convertToDouble();
    if (auto mulconstOp =
            dyn_cast_or_null<top::MulConstOp>(op.getInput().getDefiningOp())) {
      if (!mulconstOp.getOutput().hasOneUse())
        return failure();
      const_val *= mulconstOp.getConstVal().convertToDouble();
      op.setConstVal(APFloat(const_val));
      rewriter.replaceOp(mulconstOp, {mulconstOp.getInput()});
      return success();
    } else {
      return failure();
    }
    return success();
  }
};



struct RemoveMulConst : public OpRewriterPatternEx<MulConstOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  RemoveMulConst(MLIRContext *context)
      : OpRewriterPatternEx<MulConstOp>(context,"RemoveMulConst") {}
  LogicalResult matchAndRewriteImpl(MulConstOp op,
                                PatternRewriter &rewriter) const override {
    // placeholder
    double const_val = op.getConstVal().convertToDouble();
    if (const_val != 1.0) {
      return failure();
    }
    rewriter.replaceOp(op, {op.getInput()});
    return success();
  }
};


struct MulConstantFill : public OpRewriterPatternEx<MulConstOp> {
  public:
  MulConstantFill(MLIRContext *context)
      : OpRewriterPatternEx<MulConstOp>(context,"MulConstantFill") {}
  LogicalResult matchAndRewriteImpl(MulConstOp op,
                                PatternRewriter &rewriter) const override {
    double const_val = op.getConstVal().convertToDouble();
    auto input = op.getInput();
    auto constant_op = dyn_cast<top::ConstantFillOp>(input.getDefiningOp());
    if (!constant_op) {
      return failure();
    }
    if (!constant_op->hasOneUse()) {
      return failure();
    }
    double val = constant_op.getValue().convertToDouble();
    val *= const_val;
    constant_op.setValue(APFloat(val));
    constant_op->setLoc(op.getLoc());
    rewriter.replaceOp(op, {input});
    return success();
  }
};

// merge into conv or matmul
struct MergeMulConst : public OpRewriterPatternEx<MulConstOp> {
  public:
  MergeMulConst(MLIRContext *context)
      : OpRewriterPatternEx<MulConstOp>(context,"MergeMulConst") {}
  LogicalResult matchAndRewriteImpl(MulConstOp op,
                                PatternRewriter &rewriter) const override {
    double const_val = op.getConstVal().convertToDouble();
    bool do_relu = op.getDoRelu();
    auto formerOp = op.getInput().getDefiningOp();
    if (!formerOp->getResult(0).hasOneUse()) {
      return failure();
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32() && !storage_type.isF16()) {
      return failure();
    }
    if (auto convOp = dyn_cast_or_null<top::ConvOp>(formerOp)) {
      if (convOp.getKernelShape().size() != 2) {
        return failure();
      }
      if (convOp.getDoRelu() == true && const_val < 0) {
        return failure();
      }
      auto weightOp =
          dyn_cast_or_null<top::WeightOp>(convOp.getFilter().getDefiningOp());
      auto biasOp =
          dyn_cast_or_null<top::WeightOp>(convOp.getBias().getDefiningOp());
      // This judge is for fix youdao bert bf16 acc issue.
      if (weightOp && biasOp) {
        auto weight_f32 = weightOp.read_as_float();
        auto bias_f32 = biasOp.read_as_float();
        if (weight_f32->size() == bias_f32->size()) {
          bool flag = true;
          for (uint i = 0; i < weight_f32->size(); i++) {
            if (weight_f32->at(i) + bias_f32->at(i) != 0) {
              flag = false;
              break;
            }
          }
          if (flag) {
            return failure();
          }
        }
      }
    } else if (auto fcOp = dyn_cast_or_null<top::MatMulOp>(formerOp)) {
      if ((fcOp.getDoRelu() && const_val < 0) ||
          !(module::isWeight(fcOp.getRight()))) {
        return failure();
      }
    } else {
      return failure();
    }
    for (uint i = 0; i < formerOp->getNumOperands(); i++) {
      auto value = formerOp->getOperand(i);
      auto weightOp = dyn_cast_or_null<top::WeightOp>(value.getDefiningOp());
      if (!weightOp) {
        continue;
      }
      auto weight_f32 = weightOp.read_as_float();
      // std::vector<float> new_weight_f32(weight_f32->size());
      for (auto &w : *weight_f32) {
        w *= const_val;
      }
      auto weight_type = value.getType().cast<RankedTensorType>();
      auto new_weight =
          WeightOp::create_float(formerOp, "_mergeMulConst", *weight_f32,
                                 weight_type.getShape(), storage_type);
      formerOp->setOperand(i, new_weight);
    }
    formerOp->setLoc(op.getLoc());
    if (do_relu) {
      formerOp->setAttr("do_relu", rewriter.getBoolAttr(true));
    }
    rewriter.replaceOp(op, {op.getInput()});
    return success();
  }
};

// mul to large, to 10k
struct MulTooLarge : public OpRewriterPatternEx<MulConstOp> {
  public:
     MulTooLarge(MLIRContext *context)
      : OpRewriterPatternEx<MulConstOp>(context,"MulTooLarge") {}
  LogicalResult matchAndRewriteImpl(MulConstOp op,
                                PatternRewriter &rewriter) const override {
    // placeholder
    double const_val = op.getConstVal().convertToDouble();
    if (const_val >= 1e10) {
      const_val = 10000;
    } else if (const_val <= -1e10) {
      const_val = -10000;
    } else {
      return failure();
    }
    op.setConstVal(APFloat(const_val));
    return success();
  }
};

void MulConstOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<MergeContinuousMulConst, RemoveMulConst, MergeMulConst,
                 MulTooLarge, MulConstantFill>(context);
}
