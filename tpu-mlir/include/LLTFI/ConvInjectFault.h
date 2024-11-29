
#ifndef CONV_INJECT_FAULT_H
#define CONV_INJECT_FAULT_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {

/// A pass that prints the arguments of each `top.Conv` operation.
class ConvInjectFaultPass : public PassWrapper<ConvInjectFaultPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const override { return "conv_inject_fault"; }
  StringRef getDescription() const override { return "Injects Faults within the Conv layer"; }

  void runOnOperation() override;
  
};

void registerConvInjectFaultPass();

} // namespace mlir



#endif // CONV_INJECT_FAULT_H