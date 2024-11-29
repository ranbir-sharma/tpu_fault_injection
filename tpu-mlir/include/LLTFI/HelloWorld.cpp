#include "HelloWorld.h"
// #include "mlir/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

void CountAddOperationsPass::runOnOperation() {
  ModuleOp module = getOperation();
  addCount = 0;

  module.walk([&](Operation *op) {
    llvm::outs() << "Instructions 23232 is " << op->getName() << "\n";
    if (isa<arith::AddIOp>(op)) {
      addCount++;
    }
  });

  llvm::outs() << "Number of add operations: " << addCount << "\n";
}

std::unique_ptr<Pass> createCountAddOperationsPass() {
  return std::make_unique<CountAddOperationsPass>();
}

void registerCountAddOperationsPass() {
  PassRegistration<CountAddOperationsPass>();
}


} // namespace mlir

