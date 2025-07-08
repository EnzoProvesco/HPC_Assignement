#!/bin/bash

# This script processes all images in the 'reference' directory
# by adding 50%, 75%, and 90% salt-and-pepper noise to each.

SOURCE_DIR="input/reference"
NOISE_LEVELS=(50 75 90)
counter=1
# Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Directory '$SOURCE_DIR' not found."
    exit 1
fi

# Loop through all common image files in the source directory
for img_path in "$SOURCE_DIR"/*.{jpg,jpeg,png,bmp,tiff}; do
    # Check if the file exists to avoid errors if no images are foun
    if [ -f "$img_path" ]; then
	echo "Processing image ${counter}"
	if [ "$counter" -ge 37 ]; then
        	# Loop through each noise level
        	for amount in "${NOISE_LEVELS[@]}"; do
            		echo "Processing '$img_path' with $amount% noise..."
            		python3 AddS\&P.py "$img_path" "$amount"
        	done
	fi
	((counter++))
    fi
done

echo "Processing complete."
