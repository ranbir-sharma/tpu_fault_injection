#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <random>
#include <unistd.h>
#include <limits.h>
#include <fstream>
#include <cstdlib>

using namespace llvm;
namespace mlir {

/// A pass that prints the arguments of each `top.Conv` operation.
class ConvInjectFaultPass1
    : public PassWrapper<ConvInjectFaultPass1, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const override { return "inject_fault"; }
  StringRef getDescription() const override {
    return "Injects Faults within the Conv layer";
  }

  void runOnOperation() override {

    ModuleOp module = getOperation();
    OpBuilder builder(&getContext());

    int numberOfConvLayer = 0;

    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "top.Conv") {
        numberOfConvLayer++;

      }
    });

    errs() << "Total number of conv layer is " << numberOfConvLayer << "\n";

    // targetting a random conv layer    
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<int> dist(1, 53);

    // Generate the random conv layer number
    int random_number = dist(gen);
    errs() << "Printing the random conv layer number " << random_number << "\n";

    char cwd[PATH_MAX]; // PATH_MAX is defined in <limits.h>
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        errs() << "Current working directory: " << cwd << "\n";
    } else {
        perror("getcwd error"); // Print error if getcwd fails
    }

    std::string file_name = "InjectHelper.txt";

    std::ofstream file(file_name, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        errs() << "Failed to open file: " << file_name << "\n";
        return;
    }

    // Write "conv {random_number}" to the file
    file << random_number << "\n";
    file << 0 << "\n";

    // Close the file
    file.close();

    
  }
};

void registerConvInjectFaultPass1() {
  PassRegistration<ConvInjectFaultPass1>();
}

} // namespace mlir

void applyFaultToTensor(float *tensorData, size_t dataSize) {
  for (size_t i = 0; i < dataSize; ++i) {
    tensorData[i] += 1.0f; // Example fault: add a constant
  }
}