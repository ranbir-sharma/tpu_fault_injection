#include "mlir/Dialect/Arith/IR/Arith.h" // Include the Arith dialect for arith operations
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Operation.h"
// #include "mlir/IR/Module.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace {
  // Define the pass that changes add to mul.
  struct ReplaceAddWithMulPass : public PassWrapper<ReplaceAddWithMulPass, OperationPass<ModuleOp>> {
    // Constructor for the pass.
    ReplaceAddWithMulPass() = default;

    StringRef getArgument() const override { return "replace-add-with-mul"; }
    StringRef getDescription() const override { return "Replaces all add instructions with mul instructions"; }

    // Run the pass on the provided module.
    void runOnOperation() override {
      // Get the current module.
      ModuleOp module = getOperation();

      // Traverse through all operations within the module.
      module.walk([&](Operation *op) {
        // Check if the operation is an `arith.addi` operation.
        if (auto addOp = dyn_cast<arith::AddIOp>(op)) {  
          // Create a new `arith.muli` operation with the same operands.
          OpBuilder builder(addOp);
          Operation* mulOp = builder.create<arith::MulIOp>(addOp.getLoc(), addOp.getLhs(), addOp.getRhs());

          // Replace all uses of the `add` operation with the new `mul` operation.
          addOp.replaceAllUsesWith(mulOp);

          // Erase the original `add` operation.
          // addOp.erase();
        }
      });
    }
  };
}

// Register the pass.
std::unique_ptr<Pass> createReplaceAddWithMulPass() {
  return std::make_unique<ReplaceAddWithMulPass>();
}

void registerMyPass() {
    PassRegistration<ReplaceAddWithMulPass>();
}
// std::unique_ptr<Pass> mlir::createControlFlowSinkPass() {
//   return std::make_unique<ControlFlowSink>();
// }


// // Registration.
// static PassRegistration<ReplaceAddWithMulPass> pass(
//   "replace-add-with-mul",
//   "Replaces all add instructions with mul instructions"
// );