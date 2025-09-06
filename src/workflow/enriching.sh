#!/bin/bash

# Enrichment Pipeline Script with Auto-detection
# Description: Execute filter triples, structural enrichment, and feature enrichment in sequence

# =============================================================================
# Configuration Variables
# =============================================================================

# Input dataset for step 1
INITIAL_DATASET=TRIPLE_TRANS_OUTPUT_PATH_HERE.parquet

# Output directories for auto-detection
FILTER_OUTPUT_DIR="preprocess_datasets/filter_triple_datasets"
STRUCTURAL_OUTPUT_DIR="preprocess_datasets/structural_enrich_datasets"
FEATURE_OUTPUT_DIR="preprocess_datasets/feature_enrich_datasets"

# Task names
TASK_1="filter_triples"
TASK_2="structral_enrich"
TASK_3="feature_enrich"

# Common LLM Configuration
LLM_MODEL="gpt-4o-mini-2024-07-18"
OPENAI_API_KEY=YOUR_API_KEY_HERE
BASE_URL=API_BASE_URL_HERE

# Auto-detect settings
AUTO_DETECT=true

# =============================================================================
# Helper Functions
# =============================================================================

find_latest_file() {
    local dir=$1
    local pattern=$2
    find "$dir" -name "$pattern" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

# =============================================================================
# Script Execution
# =============================================================================

echo "=========================================="
echo "Enrichment Pipeline: Filter → Structural → Feature"
echo "=========================================="
echo "Auto-detect mode: $AUTO_DETECT"
echo "LLM Model: $LLM_MODEL"
echo "Base URL: $BASE_URL"
echo "API Key: ${OPENAI_API_KEY:0:10}..."
echo "=========================================="
echo ""

# Current dataset tracker
CURRENT_DATASET="$INITIAL_DATASET"

# Step 1: Filter Triples
echo "🔄 Step 1: Filter Triples"
echo "=========================================="
echo "Task: $TASK_1"
echo "Dataset: $CURRENT_DATASET"
echo "=========================================="

if [ ! -f "$CURRENT_DATASET" ]; then
    echo "❌ Error: Input dataset not found: $CURRENT_DATASET"
    exit 1
fi

echo "Starting filter triples..."
python run.py \
    --task "$TASK_1" \
    -d "$CURRENT_DATASET" \
    --llm "$LLM_MODEL" \
    --openai_api_key "$OPENAI_API_KEY" \
    --base_url "$BASE_URL"

if [ $? -ne 0 ]; then
    echo "❌ Step 1 failed: Filter triples"
    exit 1
fi

echo "✅ Step 1 completed: Filter triples"

# Auto-detect step 1 output
if [ "$AUTO_DETECT" = true ]; then
    echo "🔍 Auto-detecting output from step 1..."
    CURRENT_DATASET=$(find_latest_file "$FILTER_OUTPUT_DIR" "cwq_*_filter_triple_*.parquet")
    if [ -n "$CURRENT_DATASET" ] && [ -f "$CURRENT_DATASET" ]; then
        echo "✅ Found: $CURRENT_DATASET"
    else
        echo "❌ Could not find step 1 output file"
    fi
fi

echo ""

CURRENT_DATASET="preprocess_datasets/filter_triple_datasets/cwq_gpt-4o-mini-2024-07-18_filter_triple_2025-09-06_19-08-53.parquet"

# Step 2: Structural Enrich
echo "🔄 Step 2: Structural Enrich"
echo "=========================================="
echo "Task: $TASK_2"
echo "Dataset: $CURRENT_DATASET"
echo "=========================================="

if [ ! -f "$CURRENT_DATASET" ]; then
    echo "❌ Error: Dataset not found: $CURRENT_DATASET"
    exit 1
fi

echo "Starting structural enrichment..."
python run.py \
    --task "$TASK_2" \
    -d "$CURRENT_DATASET" \
    --llm "$LLM_MODEL" \
    --openai_api_key "$OPENAI_API_KEY" \
    --base_url "$BASE_URL"

if [ $? -ne 0 ]; then
    echo "❌ Step 2 failed: Structural enrich"
    exit 1
fi

echo "✅ Step 2 completed: Structural enrich"

# Auto-detect step 2 output
if [ "$AUTO_DETECT" = true ]; then
    echo "🔍 Auto-detecting output from step 2..."
    CURRENT_DATASET=$(find_latest_file "$STRUCTURAL_OUTPUT_DIR" "cwq_*_structural_enrich_*.parquet")
    if [ -n "$CURRENT_DATASET" ] && [ -f "$CURRENT_DATASET" ]; then
        echo "✅ Found: $CURRENT_DATASET"
    else
        echo "❌ Could not find step 2 output file"
    fi
fi

echo ""

CURRENT_DATASET="preprocess_datasets/structural_enrich_datasets/cwq_gpt-4o-mini-2024-07-18_structural_enrich_2025-09-06_19-33-38.parquet"

# Step 3: Feature Enrich
echo "🔄 Step 3: Feature Enrich"
echo "=========================================="
echo "Task: $TASK_3"
echo "Dataset: $CURRENT_DATASET"
echo "=========================================="

if [ ! -f "$CURRENT_DATASET" ]; then
    echo "❌ Error: Dataset not found: $CURRENT_DATASET"
    exit 1
fi

echo "Starting feature enrichment..."
python run.py \
    --task "$TASK_3" \
    -d "$CURRENT_DATASET" \
    --llm "$LLM_MODEL" \
    --openai_api_key "$OPENAI_API_KEY" \
    --base_url "$BASE_URL"

if [ $? -ne 0 ]; then
    echo "❌ Step 3 failed: Feature enrich"
    exit 1
fi

echo "✅ Step 3 completed: Feature enrich"
echo ""

echo "=========================================="
echo "🎉 Enrichment Pipeline completed successfully!"
echo "=========================================="
echo "✅ Step 1: Filter triples - COMPLETED"
echo "✅ Step 2: Structural enrich - COMPLETED"
echo "✅ Step 3: Feature enrich - COMPLETED"
echo "=========================================="