#!/bin/bash

#SBATCH --job-name=gen_prompt  # Set a descriptive job name
#SBATCH --partition=EM         # Set partition node
#SBATCH --nodes=1              # Request 1 node
#SBATCH --ntasks-per-node=1    # Run 1 task per node
#SBATCH --gres=gpu:4           # Run with upto 4 GPUs

# Load necessary modules (if any)
# module load <module_name>

# Function to generate the current datetime in desired format
generate_datetime() {
  date +"%Y%m%d_%H%M%S"  # Example: 20240312_112705
}

# Set variables 
BINARY_FILE_PATH=$1       # Path to your binary file
MODEL_PATH=$2             # Path to your binary file
INPUT_FOLDER_PATH=$3      # Folder with input text files (passed as argument)
OUTPUT_FOLDER_PATH=$4     # Path to your binary file

mkdir -p $OUTPUT_FOLDER_PATH

# Ensure the input folder exists
if [ ! -d "$INPUT_FOLDER_PATH" ]; then
  echo "Input folder not found: $INPUT_FOLDER_PATH"
  exit 1
fi

# Iterate through text files in the input folder
for file in "$INPUT_FOLDER_PATH"/*.txt; do
  # Split filename and extension
  filename="${file%.*}"
  extension="${file##*.}"

  # Generate datetime stamp
  datetime=$(generate_datetime)

  # Construct the out file path
  out_filepath="$OUTPUT_FOLDER_PATH/$filename_$datetime.$extension"
  echo $out_filepath

  $BINARY_FILE_PATH -m test -f "$file" -o "$out_filepath" 
done
