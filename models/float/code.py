import numpy as np

# Load the .npz file
data = np.load('resnet50_tf_bm1684x_f32_tpu_outputs.npz')

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
    
    print("Top 5 numbers:", top_numbers)
    print("Indices of top 5 numbers:", top_indices)

