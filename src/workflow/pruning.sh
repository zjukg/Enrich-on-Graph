#!/bin/bash

# Pipeline Script: Three Channel Pruning + Triple Trans (Auto-detect output)
# Description: Execute three channel pruning followed by triple transformation

# =============================================================================
# Configuration Variables
# =============================================================================

# Step 1: Three Channel Pruning Configuration
TASK_1="llm_pruning_three_channels"
DATASET_1=QUESITION_DECOMPOSE_DATASET_PATH_HERE.parquet
EMBEDDING_MODEL="sentence-transformers"
EMBEDDING_MODEL_PATH=YOUR_EMBEDDING_MODEL_PATH_HERE
PRUNING_TOP_K="100"

# Step 2: Triple Trans Configuration
TASK_2="triple_trans"
# DATASET_2 will be auto-detected or set manually
DATASET_2=THREE_CHANNEL_PRUNING_OUTPUT_PATH_HERE.parquet
LLM_MODEL="gpt-4o-mini-2024-07-18"
OPENAI_API_KEY=YOUR_API_KEY_HERE
BASE_URL=API_BASE_URL_HERE

# Auto-detect settings
AUTO_DETECT_OUTPUT=true  # Set to false if you want to use fixed DATASET_2
OUTPUT_DIR="preprocess_datasets/llm_pruning_three_channels_datasets"

# =============================================================================
# Script Execution
# =============================================================================

echo "=========================================="
echo "Pipeline Execution: Three Channel Pruning + Triple Trans"
echo "=========================================="
echo ""

# Step 1: Three Channel Pruning
echo "🔄 Step 1: Three Channel Pruning"
echo "=========================================="
echo "Task: $TASK_1"
echo "Dataset: $DATASET_1"
echo "Embedding Model: $EMBEDDING_MODEL"
echo "Embedding Model Path: $EMBEDDING_MODEL_PATH"
echo "Pruning Top K: $PRUNING_TOP_K"
echo "=========================================="

if [ ! -f "$DATASET_1" ]; then
    echo "❌ Error: Input dataset not found: $DATASET_1"
    exit 1
fi

echo "Starting three channel pruning..."
python run.py \
    --task "$TASK_1" \
    -d "$DATASET_1" \
    --embedding_model "$EMBEDDING_MODEL" \
    --embedding_model_path "$EMBEDDING_MODEL_PATH" \
    --pruning_top_k "$PRUNING_TOP_K"

if [ $? -ne 0 ]; then
    echo "❌ Step 1 failed: Three channel pruning"
    exit 1
fi

echo "✅ Step 1 completed: Three channel pruning"

# Auto-detect output file if enabled
if [ "$AUTO_DETECT_OUTPUT" = true ]; then
    echo "🔍 Auto-detecting output file from step 1..."
    LATEST_FILE=$(find "$OUTPUT_DIR" -name "cwq_sentence-transformers_*_llm_pruning_three_channels_*.parquet" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$LATEST_FILE" ] && [ -f "$LATEST_FILE" ]; then
        DATASET_2="$LATEST_FILE"
        echo "✅ Found output file: $DATASET_2"
    else
        echo "⚠️  Could not auto-detect output file, using predefined path"
    fi
fi

echo ""

# Step 2: Triple Trans
echo "🔄 Step 2: Triple Trans"
echo "=========================================="
echo "Task: $TASK_2"
echo "Dataset: $DATASET_2"
echo "LLM Model: $LLM_MODEL"
echo "Base URL: $BASE_URL"
echo "API Key: ${OPENAI_API_KEY:0:10}..."
echo "=========================================="

if [ ! -f "$DATASET_2" ]; then
    echo "❌ Error: Dataset for step 2 not found: $DATASET_2"
    exit 1
fi

echo "Starting triple transformation..."
python run.py \
    --task "$TASK_2" \
    -d "$DATASET_2" \
    --llm "$LLM_MODEL" \
    --openai_api_key "$OPENAI_API_KEY" \
    --base_url "$BASE_URL"

if [ $? -ne 0 ]; then
    echo "❌ Step 2 failed: Triple trans"
    exit 1
fi

echo "✅ Step 2 completed: Triple trans"
echo ""

echo "=========================================="
echo "🎉 Pipeline completed successfully!"
echo "=========================================="
echo "✅ Step 1: Three channel pruning - COMPLETED"
echo "✅ Step 2: Triple trans - COMPLETED"
echo "=========================================="