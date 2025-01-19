import numpy as np
import sys

# Load the .npz file
data = np.load('resnet50_tf_bm1684x_f32_tpu_outputs.npz')

filename = "failed.txt"

def writeToFile(top5Values, top5Index, layerNumber, BitNumber):
    with open(filename, "a") as file:
        file.write(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
        file.write(f"Fault in layer {layerNumber} and bit {BitNumber}\n")
        file.write(f"Top 5 values are: {top5Values}\n")
        file.write(f"Top 5 index are: {top5Index}\n")
        file.write("\n")
        

# Access command-line arguments
arguments = sys.argv[1:]  # Skip the script name

# Example usage
if len(arguments) >= 2:
    layerNumber = arguments[0]
    BitNumber = arguments[1]
else:
    print("[DEBUG] Not enough arguments provided.")
    sys.exit()

# List all keys in the file
print("Keys:", data.files)

# Access individual arrays using their keys
for key in data.files:
    # print((data[key][0]))
    array = np.array(data[key][0])
    # print(f"Data for {key}:")
    # print(data[key])
    top_indices = np.argsort(array)[-5:][::-1]
    top_numbers = array[top_indices]
    writeToFile(top_numbers, top_indices, layerNumber, BitNumber)
    
    print("Top 5 numbers:", top_numbers)
    print("Indices of top 5 numbers:", top_indices)

