#!/bin/bash

# Constants
OUT_DIR="data/topk_my"
BATCH_SIZE=256

# Check if required arguments are provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <model_name> <model_type> <k> <perturbation> <sim_fn>"
    echo "Example: $0 bert-base-uncased bi 1000 TFC1 dot"
    exit 1
fi

MODEL_NAME=$1
MODEL_TYPE=$2
K=$3
PERTURBATION=$4
SIM_FN=$5

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"


# Run the Python scriptz
python scripts/topk.py \
    --model_name_or_path="$MODEL_NAME" \
    --model_type="$MODEL_TYPE" \
    --sim_fn="$SIM_FN" \
    --in_file="data/${PERTURBATION}-data-msmarco.tsv.gz" \
    --out_path="$OUT_DIR" \
    --k="$K" \
    --batch_size="$BATCH_SIZE" \
    --perturbation_type="$PERTURBATION"
