// HelloWorld.h
#ifndef HELLO_WORLD_H
#define HELLO_WORLD_H

#include "mlir/Dialect/Arith/IR/Arith.h"  // Include the Arith dialect
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {

/// This pass counts all `arith.addi` operations in the provided MLIR file.
class CountAddOperationsPass : public PassWrapper<CountAddOperationsPass, OperationPass<ModuleOp>> {
public:
  // Constructors for the pass
  CountAddOperationsPass() = default;
  CountAddOperationsPass(const CountAddOperationsPass &pass) {}

  /// Returns the command-line argument for invoking this pass
  StringRef getArgument() const override { return "count-add-ops"; }

  /// Returns the description of this pass
  StringRef getDescription() const override { return "Counts all add operations in the MLIR file"; }

  /// Runs the pass on the provided module
  void runOnOperation() override;

private:
  int addCount = 0; // Variable to store the count of add operations
};

/// Factory function to create an instance of the CountAddOperationsPass.
std::unique_ptr<Pass> createCountAddOperationsPass();

/// Optional: Function to register this pass, if needed for pass pipelines.
void registerCountAddOperationsPass();

} // namespace mlir

#endif // HELLO_WORLD_H