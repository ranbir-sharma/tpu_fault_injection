#include "ConvInjectFault.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace mlir {


void ConvInjectFaultPass::runOnOperation() {
  ModuleOp module = getOperation();

    for (auto func : module.getOps<func::FuncOp>()) {
      llvm::outs() << "Function: " << func.getName() << "\n";

      for (Block &block : func.getBody()) {
        for (Operation &op : block) {
          // llvm::outs() << "  Operation: " << op.getName() << "\n";
          if(op.getName().getStringRef() == "top.Conv") {
            // errs() << op << "\n";
            for (auto arg : op.getOperands()) {
              // llvm::outs() << "  Argument: " << arg << "\n";
              if (auto opResult = arg.dyn_cast<mlir::OpResult>()) {
                // If the value is an operation result, get the defining operation
                mlir::Operation *defOp = opResult.getDefiningOp();
                llvm::errs() << "Operation result produced by: " << defOp->getName() << "\n";

                if(defOp->getName().getStringRef() == "top.Weight") {
                  // errs() << "Got this : " << *defOp << "\n";
                  mlir::Location loc = defOp->getLoc();
                  // llvm::outs() << "Location: " << loc << "\n";

                  if (auto nameLoc = loc.dyn_cast<mlir::NameLoc>()) {
                    // Extract and print the name as a string
                    llvm::StringRef locString = nameLoc.getName();
                    llvm::outs() << "Header name " << locString << "\n";
                  } else {
                    llvm::outs() << "Location is not a NameLoc\n";
                  }
                }
              }
            }
            return;
          }
        }
      }
    }

  // module.walk([&](Operation *op) {
  //   // Check if the operation is a `top.Conv`.
  //   if (op->getName().getStringRef() == "top.Conv") {
  //     llvm::outs() << "Found top.Conv operation with arguments:\n";
      
  //     // Iterate over each operand (argument) and print it.
  //     for (auto arg : op->getOperands()) {
  //       llvm::outs() << "  Argument: " << arg << "\n";
        
  //     }
  //     return;
  //   }
  // });
}




void registerConvInjectFaultPass() {
    PassRegistration<ConvInjectFaultPass>();

}
} // namespace mlir