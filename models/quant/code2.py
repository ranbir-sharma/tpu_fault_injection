import re
import matplotlib.pyplot as plt

# Initialize arrays to store data
faults = []  # To store layer and bit information
top_values = []  # To store top 5 values
top_indices = []  # To store top 5 indices
values = {}

for i in range(1, 54):
    values[i] = 0

values[50] = 1
values[52] = 1

 
# Open and read the file
with open("failed3.txt", "r") as file:
    content = file.read()

# Find all "Fault in layer X and bit Y" entries
fault_matches = re.findall(r"Fault in layer (\d+) and bit (\d+)", content)

# Find all "Top 5 values" entries
values_matches = re.findall(r"Top 5 values are: \[(.*?)\]", content)

# Find all "Top 5 index" entries
indices_matches = re.findall(r"Top 5 index are: \[(.*?)\]", content)


# Process and store the extracted data
for i in range(len(fault_matches)):
    layer, bit = map(int, fault_matches[i])
    faults.append((layer, bit))  # Store layer and bit as a tuple
    value = values.get(layer, 0)
    values[layer] = value + 1
    
values = dict(sorted(values.items()))
x = list(values.keys())
y = list(values.values())

print(len(x))
print(len(y))

plt.figure(figsize=(12, 6)) 
plt.bar(x, y)
plt.xticks(ticks=x, labels=x, rotation=45, ha='right')
plt.axvline(x=26, color='red', linestyle='--', linewidth=2, label='x = 26')

# Add labels and title
plt.xlabel('Convolution Layer Number')
plt.ylabel('Number of SDC Instance')
plt.title('Distribution of SDC Instances Across Convolutional Layers')
plt.tight_layout()

# Show the plot
plt.savefig('bar_graph.png')

# # Print the structured data
# print("Faults (Layer, Bit):", faults)
# print("Top 5 Values:", top_values)
# print("Top 5 Indices:", top_indices)