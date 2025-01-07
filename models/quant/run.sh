#!/bin/bash

# Function to clear a file and write three numbers
update_file() {
    # First number is that target layer 
    # Second number is the current layer, always set to 0
    # Third number is the fault bit 
    local FILE="$1"  # First parameter: file name
    local NUM1="$2"  # Second parameter: first number
    local NUM2="$3"  # Third parameter: second number
    local NUM3="$4"  # Fourth parameter: third number

    # Clear the content of the file
    > "$FILE"

    # Write the three numbers into the file
    echo "$NUM1" >> "$FILE"
    echo "$NUM2" >> "$FILE"
    echo "$NUM3" >> "$FILE"

    # Notify the user
    echo "File $FILE has been updated with $NUM1, $NUM2, and $NUM3."
}

# Example usage of the function

update_file "InjectHelper.txt" 48 0 16
failed=0


for i in $(seq 0 10000)  # Loop for 100 iterations
do
    echo "Starting $i"
    random_layer=$((RANDOM % 46))
    random_bit=$((RANDOM % 8))

    if [ $random_bit -eq 7 ]; then 
        random_bit=31
    fi

    update_file "InjectHelper.txt" $random_layer 0 $random_bit
    model_deploy.py \
    --mlir resnet50_tf.mlir \
    --quantize INT8 \
    --chip bm1684x \
    --test_input resnet50_tf_top_outputs_102.npz \
    --test_reference resnet50_tf_top_outputs_102.npz \
    --tolerance 1.0,1.0\
    --model resnet50_tf_1684x.bmodel

    if [ $? -eq 0 ]; then
        echo "The command completed successfully."
    else
        echo "The command failed with an error."
        echo "[ERROR] : IT FAILED"
        failed=$((failed + 1))
        echo "[VALUES] $random_layer $random_bit"
        python3 code.py $random_layer $random_bit
    fi
    echo "Completed $i"
done

echo "Failed count: $failed"