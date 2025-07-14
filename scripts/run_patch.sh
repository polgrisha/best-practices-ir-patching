#!/bin/bash

# Constants
OUT_DIR="data/patching_unnormalised"

# Check if required arguments are provided
if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <model_name> <model_type> <k> <perturbation> <patch_type> <sim_fn> <batch_size>"
    echo "Example: $0 bert-base-uncased bi 1000 TFC1 head_all dot 1"
    exit 1
fi

MODEL_NAME=$1
MODEL_TYPE=$2
K=$3
PERTURBATION=$4
PATCH_TYPE=$5
SIM_FN=$6
BATCH_SIZE=$7

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# convert model name by replacing / with -
FORMATTED_MODEL_NAME=$(echo $MODEL_NAME | sed 's/\//-/g')

# Run the Python script
python scripts/patch.py \
    --model_name_or_path="$MODEL_NAME" \
    --model_type="$MODEL_TYPE" \
    --sim_fn="$SIM_FN" \
    --in_file="data/topk_my/${FORMATTED_MODEL_NAME}_${MODEL_TYPE}_${PERTURBATION}_topk_${K}_msmarco.tsv" \
    --out_path="$OUT_DIR" \
    --k="$K" \
    --batch_size="$BATCH_SIZE" \
    --perturbation_type="$PERTURBATION" \
    --patch_type="$PATCH_TYPE" \