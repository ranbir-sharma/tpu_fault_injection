//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace tpu_mlir::top;

struct ConvertEinsum : public OpRewriterPatternEx<EinsumOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

    ConvertEinsum(mlir::MLIRContext *context)
      : OpRewriterPatternEx<EinsumOp>(context, "ConvertEinsum") {}

  LogicalResult matchAndRewriteImpl(EinsumOp op,
                                PatternRewriter &rewriter) const override {
    // if (op.getInputs().size() != 2 || module::isWeight(op.getInputs()[0])) {
    //   llvm_unreachable("Not support now.");
    //   // return failure();
    // }
    auto none = module::getNoneOp(op);
    auto mode = op.getMode().str();
    auto lhs = op.getInputs()[0];
    auto rhs = op.getInputs()[1];
    auto lshape = module::getShape(lhs);
    auto rshape = module::getShape(rhs);
    std::string lname = module::getName(lhs).str();
    std::string rname = module::getName(rhs).str();
    std::string name = module::getName(op.getOutput()).str();

    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    if (mode == "a,b->ab") {
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto newType = RankedTensorType::get({lshape[0], 1}, module::getElementType(lhs));
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_to2dim"));
      auto lrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});
      operands.push_back(lrsOp);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      newType = RankedTensorType::get({1, rshape[0]}, module::getElementType(rhs));
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_to2dim"));
      auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
      operands.push_back(rrsop);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      auto matmulOp = rewriter.create<MatMulOp>(op.getLoc(), op.getType(), operands, attrs);
      op.replaceAllUsesWith(matmulOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "ab,ab->a") {
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto newType = RankedTensorType::get({lshape[0], 1, lshape[1]}, module::getElementType(lhs));
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_to3dim"));
      auto lrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});
      operands.push_back(lrsOp);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      newType = RankedTensorType::get({rshape[0], rshape[1], 1}, module::getElementType(rhs));
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_to3dim"));
      auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
      operands.push_back(rrsop);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      newType = RankedTensorType::get({lshape[0], 1, 1}, module::getElementType(op));
      loc = NameLoc::get(rewriter.getStringAttr(name + "_matmul"));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      auto reshapeOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(), ValueRange{matmulOp});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "ab,acb->ac") {
      // matmul([a,1,1,b],[a,c,b,1])->[a,c,1,1]
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto newType = RankedTensorType::get({lshape[0], 1, 1, lshape[1]}, module::getElementType(lhs));
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_to4dim"));
      auto lrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});
      operands.push_back(lrsOp);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      newType = RankedTensorType::get({rshape[0], rshape[1], rshape[2], 1}, module::getElementType(rhs));
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_to4dim"));
      auto rrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
      operands.push_back(rrsOp);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);

      loc = NameLoc::get(rewriter.getStringAttr(name + "_matmul"));
      newType = RankedTensorType::get({lshape[0], rshape[1], 1, 1}, module::getElementType(op));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      auto reshapeOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(), ValueRange{matmulOp});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "ab,cdb->acd") {
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto newType = RankedTensorType::get({1, lshape[0], lshape[1]}, module::getElementType(lhs));
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_to3dim"));
      auto lrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});
      operands.push_back(lrsOp);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      newType = RankedTensorType::get({1, rshape[0] * rshape[1], rshape[2]}, module::getElementType(rhs));
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_to3dim"));
      auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_trans"));
      newType = RankedTensorType::get({1, rshape[2], rshape[0] * rshape[1]}, module::getElementType(rrsop));
      attrs.push_back(rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1})));
      auto tranOp = rewriter.create<PermuteOp>(loc, newType, ValueRange{rrsop}, attrs);
      operands.push_back(tranOp);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);

      loc = NameLoc::get(rewriter.getStringAttr(name + "_matmul"));
      newType = RankedTensorType::get({1, lshape[0],  rshape[0] * rshape[1]}, module::getElementType(op));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      auto reshapeOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(), ValueRange{matmulOp});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abc,db->adc") {
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      auto newType = RankedTensorType::get({1, rshape[0], rshape[1]}, module::getElementType(rhs));
      auto loc = NameLoc::get(rewriter.getStringAttr(rname + "_to3dim"));
      auto rrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
      operands.push_back(rrsOp);
      operands.push_back(lhs);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      auto matmulOp = rewriter.create<MatMulOp>(op.getLoc(), op.getType(), operands, attrs);
      op.replaceAllUsesWith(matmulOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,abce->acde") {
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_permute"));
      auto newType = RankedTensorType::get({lshape[0], lshape[2], lshape[3], lshape[1]}, module::getElementType(lhs));
      attrs.push_back(rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 3, 1})));
      auto ltranOp = rewriter.create<PermuteOp>(loc, newType, ValueRange{lhs}, attrs);
      operands.push_back(ltranOp);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      attrs.clear();
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_permute"));
      newType = RankedTensorType::get({rshape[0], rshape[2], lshape[1], rshape[3]}, module::getElementType(rhs));
      attrs.push_back(rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
      auto rtranOp = rewriter.create<PermuteOp>(loc, newType, ValueRange{rhs}, attrs);
      operands.push_back(rtranOp);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      attrs.clear();
      auto matmulOp = rewriter.create<MatMulOp>(op.getLoc(), op.getType(), operands, attrs);
      op.replaceAllUsesWith(matmulOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode ==  "abcd,acd->abc") {
      // TODO : check whether tile can be optimized
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto newType = RankedTensorType::get({lshape[0] * lshape[1] * lshape[2], 1, lshape[3]}, module::getElementType(lhs));
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_to3dim"));
      auto lrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});
      operands.push_back(lrsOp);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      newType = RankedTensorType::get({rshape[0], 1, rshape[1], rshape[2], 1}, module::getElementType(rhs));
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_to5dim"));
      auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
      newType = RankedTensorType::get({rshape[0], lshape[1], rshape[1], rshape[2], 1}, module::getElementType(rhs));
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_tile"));
      attrs.push_back(rewriter.getNamedAttr("tile", rewriter.getI64ArrayAttr({1, lshape[1], 1, 1, 1})));
      auto tileOp = rewriter.create<TileOp>(loc, newType, ValueRange{rrsop}, attrs);
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_to3dim"));
      newType = RankedTensorType::get({rshape[0] * lshape[1] * rshape[1], rshape[2], 1}, module::getElementType(rhs));
      auto reshapeOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{tileOp});
      operands.push_back(reshapeOp);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      attrs.clear();
      loc = NameLoc::get(rewriter.getStringAttr(name + "_matmul"));
      newType = RankedTensorType::get({lshape[0] * lshape[1] * lshape[2], 1, 1}, module::getElementType(op));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      reshapeOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(), ValueRange{matmulOp});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,cde->abe") {
      // lhs_reshape_rst = [lhs_shape[0] * lhs_shape[1], lhs_shape[2] * lhs_shape[3]]
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto newType = RankedTensorType::get({lshape[0] * lshape[1], lshape[2] * lshape[3]}, module::getElementType(lhs));
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_to2dim"));
      auto lrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});
      operands.push_back(lrsOp);
      newType = RankedTensorType::get({rshape[0] * rshape[1], rshape[2]}, module::getElementType(rhs));
      if (module::isWeight(rhs)) {
        rhs.setType(newType);
        operands.push_back(rhs);
      } else {
        rewriter.setInsertionPointAfter(rhs.getDefiningOp());
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_to2dim"));
        auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
        operands.push_back(rrsop);
      }
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      newType = RankedTensorType::get({lshape[0] * lshape[1], rshape[2]}, module::getElementType(op));
      loc = NameLoc::get(rewriter.getStringAttr(name + "_matmul"));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      auto orsOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(), ValueRange{matmulOp});
      op.replaceAllUsesWith(orsOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,bed->abce") {
      // TODO ： remove right_transpose to meet sg2380 gdma update
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      // batch matmul does not support broadcast
      // temporary solution
      // [h, k, c] -> [1, h, k, c] -> [b, h, k, c]
      operands.push_back(lhs);
      RankedTensorType newType;
      if (auto wOp = dyn_cast<top::WeightOp>(rhs.getDefiningOp())) {
        auto storage_type = module::getStorageType(rhs);
        assert(storage_type.isF32() && "Todo, supoort more weight type");
        auto data = wOp.read_as_byte();
        uint8_t *dptr;
        newType = RankedTensorType::get({lshape[0], rshape[0], rshape[1], rshape[2]}, module::getElementType(rhs));
        std::vector<float_t> new_filter(newType.getNumElements(), 0);
        dptr = (uint8_t *)new_filter.data();
        for (int32_t i = 0; i < lshape[0]; i++) {
          auto offset = i * data->size();
          memcpy(dptr + offset, data->data(), data->size());
        }
        auto new_op = top::WeightOp::create(op, "folder", new_filter, newType);
        wOp.replaceAllUsesWith(new_op.getDefiningOp());
        operands.push_back(new_op);
        rewriter.eraseOp(wOp);
      } else {
        auto loc = NameLoc::get(rewriter.getStringAttr(rname + "_reshape"));
        newType = RankedTensorType::get({1, rshape[0], rshape[1], rshape[2]}, module::getElementType(rhs));
        auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
        newType = RankedTensorType::get({lshape[0], rshape[0], rshape[1], rshape[2]}, module::getElementType(rhs));
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_tile"));
        attrs.push_back(rewriter.getNamedAttr("tile", rewriter.getI64ArrayAttr({lshape[0], 1, 1, 1})));
        auto tileOp = rewriter.create<TileOp>(loc, newType, ValueRange{rrsop}, attrs);
        attrs.clear();
        operands.push_back(tileOp);
      }
      operands.push_back(none);
      // [b, h, w, c] * [b, h, k, c]^T -> [b, h, w, k]
      attrs.push_back(rewriter.getNamedAttr("right_transpose", rewriter.getBoolAttr(true)));
      rewriter.setInsertionPoint(op);
      auto matmulOp = rewriter.create<MatMulOp>(op.getLoc(), op.getType(), operands, attrs);
      op.replaceAllUsesWith(matmulOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,ced->abce") {
      // dumb implementation
      // [b, h, w, c] -> [b, w, h, c]
      // trans_shape = [lhs_shape[0], lhs_shape[2], lhs_shape[1], lhs_shape[3]]
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_trans"));
      auto newType = RankedTensorType::get({lshape[0], lshape[2], lshape[1], lshape[3]}, module::getElementType(lhs));
      attrs.push_back(rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
      auto tranOp = rewriter.create<PermuteOp>(loc, newType, ValueRange{lhs}, attrs);
      attrs.clear();
      operands.push_back(tranOp);
      // [w, k, c] -> [1, w, k, c] -> [b, w, k, c]
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      if (auto wOp = dyn_cast<top::WeightOp>(rhs.getDefiningOp())) {
        auto storage_type = module::getStorageType(rhs);
        assert(storage_type.isF32() && "Todo, supoort more weight type");
        auto data = wOp.read_as_byte();
        uint8_t *dptr;
        newType = RankedTensorType::get({lshape[0], rshape[0], rshape[1], rshape[2]}, module::getElementType(rhs));
        std::vector<float_t> new_filter(newType.getNumElements(), 0);
        dptr = (uint8_t *)new_filter.data();
        for (int32_t i = 0; i < lshape[0]; i++) {
          auto offset = i * data->size();
          memcpy(dptr + offset, data->data(), data->size());
        }
        auto new_op = top::WeightOp::create(op, "folder", new_filter, newType);
        wOp.replaceAllUsesWith(new_op.getDefiningOp());
        operands.push_back(new_op);
        rewriter.eraseOp(wOp);
      } else {
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_reshape"));
        newType = RankedTensorType::get({1, rshape[0], rshape[1], rshape[2]}, module::getElementType(rhs));
        auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_tile"));
        attrs.push_back(rewriter.getNamedAttr("tile", rewriter.getI64ArrayAttr({lshape[0], 1, 1, 1})));
        newType = RankedTensorType::get({lshape[0], rshape[0], rshape[1], rshape[2]}, module::getElementType(rhs));
        auto tileOp = rewriter.create<TileOp>(loc, newType, ValueRange{rrsop}, attrs);
        attrs.clear();
        operands.push_back(tileOp);
      }
      operands.push_back(none);
      // [b, w, h, c] * [b, w, k, c]^T -> [b, w, h, k]
      newType = RankedTensorType::get({lshape[0], lshape[2], lshape[1], rshape[1]}, module::getElementType(op));
      attrs.push_back(rewriter.getNamedAttr("right_transpose", rewriter.getBoolAttr(true)));
      rewriter.setInsertionPoint(op);
      loc = NameLoc::get(rewriter.getStringAttr(name + "_matmul"));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      attrs.clear();
      // [b, w, h, k] -> [b, h, w, k]
      attrs.push_back(rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
      auto tranBackOp = rewriter.create<PermuteOp>(op.getLoc(), op.getType(), ValueRange{matmulOp}, attrs);
      op.replaceAllUsesWith(tranBackOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,abed->abce" || mode == "abcd,abde->abce") {
      // lhs(abcd) * rhs(abed)^T -> abce
      // lhs(abcd) * rhs(abde) -> abce

      auto newType = RankedTensorType::get({lshape[0], lshape[1], lshape[2], rshape[2]}, module::getElementType(op));
      if (mode == "abcd,abde->abce"){
        newType = RankedTensorType::get({lshape[0], lshape[1], lshape[2], rshape[3]}, module::getElementType(op));
      }
      rewriter.setInsertionPoint(op);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      operands.push_back(lhs);
      operands.push_back(rhs);
      operands.push_back(none);
      if (mode == "abcd,abed->abce"){
        //rhs(abed)^T
        attrs.push_back(rewriter.getNamedAttr("right_transpose", rewriter.getBoolAttr(true)));
      }

      auto loc = NameLoc::get(rewriter.getStringAttr(name));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      op.replaceAllUsesWith(matmulOp.getOperation());
      attrs.clear();
      rewriter.eraseOp(op);

    } else if (mode == "abc,adc->abd") {

      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      auto type_permute = RankedTensorType::get({rshape[0], rshape[2], rshape[1]}, module::getElementType(rhs));
      auto loc_permute = NameLoc::get(rewriter.getStringAttr(rname + "_permute"));
      attrs.push_back(rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1})));
      auto permuteOp = rewriter.create<PermuteOp>(loc_permute, type_permute, ValueRange{rhs}, attrs);
      attrs.clear();

      rewriter.setInsertionPointAfter(op);
      auto loc = NameLoc::get(rewriter.getStringAttr(name));

      auto newType = RankedTensorType::get({lshape[0], lshape[1], rshape[1]}, module::getElementType(op));
      operands.push_back(lhs);
      operands.push_back(permuteOp);
      operands.push_back(none);
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands);
      op.replaceAllUsesWith(matmulOp.getOperation());
      rewriter.eraseOp(op);

    } else if (mode == "abc,adc->adb") {
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_trans"));
      auto newType = RankedTensorType::get({lshape[0], lshape[2], lshape[1]}, module::getElementType(lhs));
      attrs.push_back(rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1})));
      auto tranOp = rewriter.create<PermuteOp>(loc, newType, ValueRange{lhs}, attrs);
      operands.push_back(rhs);
      operands.push_back(tranOp);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      auto matmulOp = rewriter.create<MatMulOp>(op.getLoc(), op.getType(), operands, attrs);
      op.replaceAllUsesWith(matmulOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abc,abd->acd") {
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_trans"));
      auto newType = RankedTensorType::get({lshape[0], lshape[2], lshape[1]}, module::getElementType(lhs));
      attrs.push_back(rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1})));
      auto tranOp = rewriter.create<PermuteOp>(loc, newType, ValueRange{lhs}, attrs);
      operands.push_back(tranOp);
      operands.push_back(rhs);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      auto matmulOp = rewriter.create<MatMulOp>(op.getLoc(), op.getType(), operands, attrs);
      op.replaceAllUsesWith(matmulOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,aecd->aeb") {
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto lreshape_loc = NameLoc::get(rewriter.getStringAttr(lname + "_reshape"));
      auto lreshape_type = RankedTensorType::get({lshape[0], lshape[1], lshape[2]*lshape[3]}, module::getElementType(lhs));
      auto lreshape_op = rewriter.create<top::ReshapeOp>(lreshape_loc, lreshape_type, ValueRange{lhs});
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      auto rreshape_loc = NameLoc::get(rewriter.getStringAttr(rname + "_reshape"));
      auto rreshape_type = RankedTensorType::get({rshape[0], rshape[1], rshape[2]*rshape[3]}, module::getElementType(rhs));
      auto rreshape_op = rewriter.create<top::ReshapeOp>(rreshape_loc, rreshape_type, ValueRange{rhs});

      rewriter.setInsertionPointAfter(lreshape_op);
      auto type_permute = RankedTensorType::get({lshape[0], lshape[2]*lshape[3], lshape[1]}, module::getElementType(lhs));
      auto loc_permute = NameLoc::get(rewriter.getStringAttr(lname + "_permute"));
      attrs.push_back(rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1})));
      auto permuteOp = rewriter.create<PermuteOp>(loc_permute, type_permute, ValueRange{lreshape_op}, attrs);
      attrs.clear();

      rewriter.setInsertionPointAfter(op);
      auto loc = NameLoc::get(rewriter.getStringAttr(name));
      auto newType = RankedTensorType::get({lshape[0], rshape[1], lshape[1]}, module::getElementType(op));
      operands.push_back(rreshape_op);
      operands.push_back(permuteOp);
      operands.push_back(none);
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands);
      op.replaceAllUsesWith(matmulOp.getOperation());
      rewriter.eraseOp(op);

    } else if (mode == "abc,cde->abde") {
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto lreshape_loc = NameLoc::get(rewriter.getStringAttr(lname + "_reshape"));
      auto lreshape_type = RankedTensorType::get({lshape[0]*lshape[1], lshape[2]}, module::getElementType(lhs));
      auto lreshape_op = rewriter.create<top::ReshapeOp>(lreshape_loc, lreshape_type, ValueRange{lhs});
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      auto rreshape_loc = NameLoc::get(rewriter.getStringAttr(rname + "_reshape"));
      auto rreshape_type = RankedTensorType::get({rshape[0], rshape[1]*rshape[2]}, module::getElementType(rhs));
      auto rreshape_op = rewriter.create<top::ReshapeOp>(rreshape_loc, rreshape_type, ValueRange{rhs});

      rewriter.setInsertionPointAfter(op);
      auto matmul_loc = NameLoc::get(rewriter.getStringAttr(name + "matmul"));
      auto matmul_type = RankedTensorType::get({lshape[0]*lshape[1], rshape[1]*rshape[2]}, module::getElementType(op));
      operands.push_back(lreshape_op);
      operands.push_back(rreshape_op);
      operands.push_back(none);
      auto matmulOp = rewriter.create<MatMulOp>(matmul_loc, matmul_type, operands);
      auto loc = NameLoc::get(rewriter.getStringAttr(name));
      auto newType = RankedTensorType::get({lshape[0], lshape[1], rshape[1], rshape[2]}, module::getElementType(matmulOp));
      auto reshape_op = rewriter.create<top::ReshapeOp>(loc, newType, ValueRange{matmulOp});
      op.replaceAllUsesWith(reshape_op.getOperation());
      rewriter.eraseOp(op);

    } else if (mode == "abcd,cde->abce") {

      // lhs :
      //     abcd -> acbd(pemute)
      // rhs :
      //     cde  -> 1cde(reshape)
      //     acde -> acde(tile)
      // matmul:
      //   lhs(acbd) * rhs(acde) = result(acbe)
      // result:
      //     acbe -> abce(pemute)
      // success!

      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_trans"));
      auto newType = RankedTensorType::get({lshape[0], lshape[2], lshape[1], lshape[3]}, module::getElementType(lhs));
      attrs.push_back(rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
      auto tranOp = rewriter.create<PermuteOp>(loc, newType, ValueRange{lhs}, attrs);
      attrs.clear();
      operands.push_back(tranOp);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      if (auto wOp = dyn_cast<top::WeightOp>(rhs.getDefiningOp())) {

        auto data = wOp.read_as_byte();
        uint8_t *dptr;
        newType = RankedTensorType::get({lshape[0], rshape[0], rshape[1], rshape[2]}, module::getElementType(rhs));
        std::vector<float_t> new_filter(newType.getNumElements(), 0);
        dptr = (uint8_t *)new_filter.data();
        for (int32_t i = 0; i < lshape[0]; i++) {
          auto offset = i * data->size();
          memcpy(dptr + offset, data->data(), data->size());
        }
        auto new_op = top::WeightOp::create(op, "folder", new_filter, newType);
        wOp.replaceAllUsesWith(new_op.getDefiningOp());
        operands.push_back(new_op);
        rewriter.eraseOp(wOp);
      } else {
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_reshape"));
        newType = RankedTensorType::get({1, rshape[0], rshape[1], rshape[2]}, module::getElementType(rhs));
        auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_tile"));
        attrs.push_back(rewriter.getNamedAttr("tile", rewriter.getI64ArrayAttr({lshape[0], 1, 1, 1})));
        newType = RankedTensorType::get({lshape[0], rshape[0], rshape[1], rshape[2]}, module::getElementType(rhs));
        auto tileOp = rewriter.create<TileOp>(loc, newType, ValueRange{rrsop}, attrs);
        attrs.clear();
        operands.push_back(tileOp);
      }
      operands.push_back(none);
      newType = RankedTensorType::get({lshape[0], lshape[2], lshape[1], rshape[2]}, module::getElementType(op));
      rewriter.setInsertionPoint(op);
      loc = NameLoc::get(rewriter.getStringAttr(name + "_matmul"));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      attrs.clear();
      attrs.push_back(rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
      auto tranBackOp = rewriter.create<PermuteOp>(op.getLoc(), op.getType(), ValueRange{matmulOp}, attrs);
      op.replaceAllUsesWith(tranBackOp.getOperation());
      rewriter.eraseOp(op);

    } else if (mode == "abc,acde->abde") {
      // lhs :
      //     abc
      // rhs :
      //     acde -> ac(de)(reshape)
      // matmul:
      //   lhs(abc) * rhs(ac(de)) = result(ab(de))
      // result:
      //     ab(de) -> abde(reshape)
      // success!
      operands.push_back(lhs);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      auto newType = RankedTensorType::get({rshape[0], rshape[1], rshape[2] * rshape[3]}, module::getElementType(op));
      if (module::isWeight(rhs)) {
        rhs.setType(newType);
        operands.push_back(rhs);
      } else {
        auto loc = NameLoc::get(rewriter.getStringAttr(rname + "_to3dim"));
        auto rhsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
        operands.push_back(rhsOp);
      }
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      newType = RankedTensorType::get({lshape[0], lshape[1], rshape[2] * rshape[3]}, module::getElementType(op));
      auto loc = NameLoc::get(rewriter.getStringAttr(name + "_matmul"));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      auto reshapeOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(), ValueRange{matmulOp});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abc,abde->acde") {
      // lhs :
      //     abc -> acb(permute)
      // rhs :
      //     abde -> ab(de)(reshape)
      // matmul:
      //   lhs(acb) * rhs(ab(de)) = result(ac(de))
      // result:
      //     ab(de) -> abde(reshape)
      // success!
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_trans"));
      auto newType = RankedTensorType::get({lshape[0], lshape[2], lshape[1]}, module::getElementType(lhs));
      attrs.push_back(rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1})));
      auto tranOp = rewriter.create<PermuteOp>(loc, newType, ValueRange{lhs}, attrs);
      operands.push_back(tranOp.getOutput());
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      newType = RankedTensorType::get({rshape[0], rshape[1], rshape[2] * rshape[3]}, module::getElementType(op));
      if (module::isWeight(rhs)) {
        rhs.setType(newType);
        operands.push_back(rhs);
      } else {
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_to3dim"));
        auto rhsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
        operands.push_back(rhsOp);
      }
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      newType = RankedTensorType::get({lshape[0], lshape[2], rshape[2] * rshape[3]}, module::getElementType(op));
      loc = NameLoc::get(rewriter.getStringAttr(name + "_matmul"));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      auto reshapeOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(), ValueRange{matmulOp});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abc,bd->abcd") {

      // lhs :
      //     abc -> abc1(reshape)
      // rhs :
      //     bd -> 1b1d(reshape)
      //     1b1d -> ab1d(tile)
      // matmul:
      //   lhs(abc1) * rhs(ab1d) = result(abcd)
      // success!

      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_to4dim"));
      auto newType = RankedTensorType::get({lshape[0], lshape[1], lshape[2], 1}, module::getElementType(op));
      auto lhsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});
      operands.push_back(lhsOp.getOutput());

      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_to4dim"));
      newType = RankedTensorType::get({1, rshape[0], 1, rshape[1]}, module::getElementType(op));
      auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});

      loc = NameLoc::get(rewriter.getStringAttr(rname + "_tile"));
      attrs.push_back(rewriter.getNamedAttr("tile", rewriter.getI64ArrayAttr({lshape[0], 1, 1, 1})));
      newType = RankedTensorType::get({lshape[0], rshape[0], 1, rshape[1]}, module::getElementType(rhs));
      auto rhs_tileOp = rewriter.create<TileOp>(loc, newType, ValueRange{rrsop}, attrs);
      attrs.clear();
      operands.push_back(rhs_tileOp);
      operands.push_back(none);
      rewriter.setInsertionPoint(op);
      newType = RankedTensorType::get({lshape[0], lshape[1], lshape[2], rshape[1]}, module::getElementType(op));
      loc = NameLoc::get(rewriter.getStringAttr(name));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      auto reshapeOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(), ValueRange{matmulOp});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abc,abc->ab") {

      // einsum('abc, abc -> ab', L, H) => sum(L * H, dim = -1)
      rewriter.setInsertionPointAfter(op);
      auto loc = NameLoc::get(rewriter.getStringAttr(name + "_Mul"));
      auto newType = RankedTensorType::get({lshape[0], lshape[1], lshape[2]}, module::getElementType(op));
      std::vector<Value> mul_operands;
      mul_operands.clear();
      mul_operands.push_back(lhs);
      mul_operands.push_back(rhs);
      auto mul_op = rewriter.create<MulOp>(loc, newType,
                                        mul_operands, attrs);

      rewriter.setInsertionPointAfter(mul_op);
      loc = NameLoc::get(rewriter.getStringAttr(name));

      newType = RankedTensorType::get({lshape[0], lshape[1]}, module::getElementType(op));
      attrs.clear();
      attrs.push_back(rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr({2})));
      attrs.push_back(rewriter.getNamedAttr("keepdims", rewriter.getBoolAttr(false)));
      attrs.push_back(rewriter.getNamedAttr("mode", rewriter.getStringAttr("Reducesum")));
      auto sumOp = rewriter.create<ReduceOp>(loc, newType, mul_op.getOutput(), attrs);
      op.getOutput().replaceAllUsesWith(sumOp);
      rewriter.eraseOp(op);
    } else if (mode == "abc,abdc,abc->abcd") {

      // einsum('abc, abdc, abc -> abcd', L, N, D) => (L.unsqueeze(2) * N * D.unsqueeze(2)).permute(0,1,3,2)
      //
      // lhs :
      //     abc -> ab1c(reshape)
      // rhs :
      //     do nothing
      // dhs:
      //     abc -> ab1c(reshape)
      //
      // lhs(ab1c) * rhs(abdc) * dhs(ab1c) => result0(abdc).permute(0,1,3,2) ==> result1(abcd)
      // success!

      auto dhs = op.getInputs()[2];
      auto dshape = module::getShape(dhs);
      std::string dname = module::getName(dhs).str();

      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_to4dim"));
      auto newType = RankedTensorType::get({lshape[0], lshape[1], 1, lshape[2]}, module::getElementType(op));
      auto lhsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});

      rewriter.setInsertionPointAfter(dhs.getDefiningOp());
      loc = NameLoc::get(rewriter.getStringAttr(dname + "_to4dim"));
      newType = RankedTensorType::get({dshape[0], dshape[1], 1, dshape[2]}, module::getElementType(op));
      auto dhsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{dhs});

      rewriter.setInsertionPointAfter(op);
      loc = NameLoc::get(rewriter.getStringAttr(name + "_LN_Mul"));
      std::vector<Value> mul_operands;
      mul_operands.push_back(lhsOp.getOutput());
      mul_operands.push_back(rhs);
      std::vector<NamedAttribute> attrs;
      newType = RankedTensorType::get({rshape[0], rshape[1], rshape[2], rshape[3]}, module::getElementType(op));
      auto mul_op = rewriter.create<MulOp>(loc, newType,
                                        mul_operands, attrs);

      rewriter.setInsertionPointAfter(mul_op);
      loc = NameLoc::get(rewriter.getStringAttr(name + "_ND_Mul"));
      mul_operands.clear();
      mul_operands.push_back(mul_op.getOutput());
      mul_operands.push_back(dhsOp.getOutput());
      auto mul_op2 = rewriter.create<MulOp>(loc, newType,
                                        mul_operands, attrs);

      rewriter.setInsertionPointAfter(mul_op2);
      loc = NameLoc::get(rewriter.getStringAttr(name));
      newType = RankedTensorType::get({rshape[0], rshape[1], rshape[3], rshape[2]}, module::getElementType(op));
      attrs.clear();
      attrs.push_back(rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 1, 3, 2})));
      auto tranOp = rewriter.create<PermuteOp>(loc, newType, mul_op2.getOutput(), attrs);
      op.getOutput().replaceAllUsesWith(tranOp);
      rewriter.eraseOp(op);
    } else if (mode == "abcd,acde,abc->abce") {
      // TODO : check whether tile can be optimized
      auto dhs = op.getInputs()[2];
      auto dshape = module::getShape(dhs);
      std::string dname = module::getName(dhs).str();

      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto newType = RankedTensorType::get({lshape[0] * lshape[1] * lshape[2], 1, lshape[3]}, module::getElementType(lhs));
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_to3dim"));
      auto lrsOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{lhs});
      operands.push_back(lrsOp);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      newType = RankedTensorType::get({rshape[0], 1, rshape[1], rshape[2], rshape[3]}, module::getElementType(rhs));
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_to5dim"));
      auto rrsop = rewriter.create<ReshapeOp>(loc, newType, ValueRange{rhs});
      newType = RankedTensorType::get({rshape[0], lshape[1], rshape[1], rshape[2], rshape[3]}, module::getElementType(rhs));
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_tile"));
      attrs.push_back(rewriter.getNamedAttr("tile", rewriter.getI64ArrayAttr({1, lshape[1], 1, 1, 1})));
      auto tileOp = rewriter.create<TileOp>(loc, newType, ValueRange{rrsop}, attrs);
      loc = NameLoc::get(rewriter.getStringAttr(rname + "_to3dim"));
      newType = RankedTensorType::get({rshape[0] * lshape[1] * rshape[1], rshape[2], rshape[3]}, module::getElementType(rhs));
      auto reshapeOp = rewriter.create<ReshapeOp>(loc, newType, ValueRange{tileOp});
      operands.push_back(reshapeOp);
      operands.push_back(none);

      rewriter.setInsertionPoint(op);
      attrs.clear();
      loc = NameLoc::get(rewriter.getStringAttr(name + "_matmul"));
      newType = RankedTensorType::get({lshape[0] * lshape[1] * lshape[2], 1, rshape[3]}, module::getElementType(op));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);

      rewriter.setInsertionPointAfter(matmulOp);
      loc = NameLoc::get(rewriter.getStringAttr(name + "_reshape"));
      newType = RankedTensorType::get({lshape[0], lshape[1], lshape[2], rshape[3]}, module::getElementType(op));
      auto matmul_reshape_op = rewriter.create<ReshapeOp>(loc, newType, ValueRange{matmulOp});

      // rewriter.setInsertionPointAfter(dhs);
      newType = RankedTensorType::get({dshape[0], dshape[1], dshape[2], 1}, module::getElementType(dhs));
      loc = NameLoc::get(rewriter.getStringAttr(dname + "_to4dim"));
      auto dhs_reshape_op = rewriter.create<ReshapeOp>(loc, newType, ValueRange{dhs});

      // rewriter.setInsertionPointAfter(matmul_reshape_op);
      operands.clear();
      operands.push_back(matmul_reshape_op.getOutput());
      operands.push_back(dhs_reshape_op.getOutput());
      auto mul_op = rewriter.create<MulOp>(op.getLoc(), op.getType(), operands, attrs);
      op.replaceAllUsesWith(mul_op.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcd,aeb->aecd"){
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto lreshape_loc = NameLoc::get(rewriter.getStringAttr(lname + "_reshape"));
      auto lreshape_type = RankedTensorType::get({lshape[0], lshape[1], lshape[2]*lshape[3]}, module::getElementType(lhs));
      auto lreshape_op = rewriter.create<top::ReshapeOp>(lreshape_loc, lreshape_type, ValueRange{lhs});

      rewriter.setInsertionPointAfter(op);
      auto newType = RankedTensorType::get({rshape[0], rshape[1], lshape[2]*lshape[3]}, module::getElementType(op));
      operands.push_back(rhs);
      operands.push_back(lreshape_op);
      operands.push_back(none);
      auto loc = NameLoc::get(rewriter.getStringAttr(name + "_matmul"));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands);
      auto reshapeOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(), ValueRange{matmulOp});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else if (mode == "abcde,afbc->abdef"){
      // TODO : top/tpu matmul inference set left_transpose false by defalut, maybe can support true
      rewriter.setInsertionPointAfter(lhs.getDefiningOp());
      auto lreshape_loc = NameLoc::get(rewriter.getStringAttr(lname + "_reshape"));
      auto lreshape_type = RankedTensorType::get({lshape[0], lshape[1], lshape[2], lshape[3]*lshape[4]}, module::getElementType(lhs));
      auto lreshape_op = rewriter.create<top::ReshapeOp>(lreshape_loc, lreshape_type, ValueRange{lhs});
      auto loc = NameLoc::get(rewriter.getStringAttr(lname + "_trans"));
      auto newType = RankedTensorType::get({lshape[0], lshape[1], lshape[3]*lshape[4], lshape[2]}, module::getElementType(lreshape_op));
      attrs.push_back(rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 1, 3, 2})));
      auto ltranOp = rewriter.create<PermuteOp>(loc, newType, ValueRange{lreshape_op}, attrs);
      attrs.clear();
      operands.push_back(ltranOp);
      // operands.push_back(lreshape_op);
      rewriter.setInsertionPointAfter(rhs.getDefiningOp());
      // if (auto wOp = dyn_cast<top::WeightOp>(rhs.getDefiningOp())) {
      //   // TODO: check weight type other F32 can satisfy
      //   auto storage_type = module::getStorageType(rhs);
      //   assert(storage_type.isF32() && "Todo, supoort more weight type");
      //   auto data = wOp.read_as_byte();
      //   uint8_t *dptr;
      //   newType = RankedTensorType::get({rshape[0], rshape[2], rshape[1], rshape[3]}, module::getElementType(rhs));
      //   std::vector<float_t> new_filter(newType.getNumElements(), 0);
      //   dptr = (uint8_t *)new_filter.data();
      //   // TODO : use continious data copy
      //   for (int32_t i = 0; i < rshape[0]; i++) {
      //       for (int32_t j = 0; j < rshape[2]; j++) {
      //           for (int32_t k = 0; k < rshape[1]; k++) {
      //               for (int32_t l = 0; l < rshape[3]; l++) {
      //                   auto offset = (i * rshape[2] * rshape[1] * rshape[3] + j * rshape[1] * rshape[3] + k * rshape[3] + l) * data->size();
      //                   memcpy(dptr + offset, data->data(), data->size());
      //               }
      //           }
      //       }
      //   }

      //   auto new_op = top::WeightOp::create(op, "folder", new_filter, newType);
      //   wOp.replaceAllUsesWith(new_op.getDefiningOp());
      //   operands.push_back(new_op);
      //   rewriter.eraseOp(wOp);
      // } else {
        loc = NameLoc::get(rewriter.getStringAttr(rname + "_trans"));
        newType = RankedTensorType::get({rshape[0], rshape[2], rshape[3], rshape[1]}, module::getElementType(rhs));
        attrs.push_back(rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 3, 1})));
        auto rtranOp = rewriter.create<PermuteOp>(loc, newType, ValueRange{rhs}, attrs);
        attrs.clear();
        operands.push_back(rtranOp);
      // }
      operands.push_back(none);
      newType = RankedTensorType::get({lshape[0], lshape[1], lshape[3] * lshape[4], rshape[1]}, module::getElementType(op));
      rewriter.setInsertionPoint(op);
      loc = NameLoc::get(rewriter.getStringAttr(name + "_matmul"));
      auto matmulOp = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
      auto reshapeOp = rewriter.create<ReshapeOp>(op.getLoc(), op.getType(), ValueRange{matmulOp});
      op.replaceAllUsesWith(reshapeOp.getOperation());
      rewriter.eraseOp(op);
    } else {
      llvm_unreachable("Einsum not support this mode now");
    }
    return success();
  }
};


void EinsumOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<ConvertEinsum>(context);
}
