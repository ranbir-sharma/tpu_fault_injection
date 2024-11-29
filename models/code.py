import numpy as np

# Load the .npz file
data = np.load('sample_output.npz')
# data = np.load('resnet50_tf_top_quantized_all_weight.npz')

# header = 'resnet50/conv1_pad/Pad;StatefulPartitionedCall/resnet50/conv1_pad/Pad'

# List all the arrays stored in the file
print("Keys in the .npz file:", data.files)

# Access and print each array
for key in data.files:
    # print(f"{key}: {data[key]}")
    # print(f"{data[header]}")
    arr = data[key]

    # Get the maximum value
    max_value = np.max(arr)

    # Get the index of the maximum value
    max_index = np.argmax(arr)

    print("Max value:", max_value)
    print("Position of max value:", max_index)