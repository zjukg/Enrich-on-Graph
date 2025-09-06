#!/bin/bash

# Query Decompose Task Script
# Description: Run query decomposition task with predefined parameters

# =============================================================================
# Configuration Variables
# =============================================================================

# Task configuration
TASK="query_decompose"

# Dataset path
DATASET_PATH=DATASET_PATH_HERE.parquet

# LLM configuration
LLM_MODEL="gpt-4o-mini-2024-07-18"
OPENAI_API_KEY=YOUR_API_KEY_HERE
BASE_URL=API_BASE_URL_HERE

# Optional: Resume path (uncomment and set if needed)
# RESUME_PATH="/path/to/resume/file.parquet"

# =============================================================================
# Script Execution
# =============================================================================

echo "=========================================="
echo "Query Decompose Task"
echo "=========================================="
echo "Task: $TASK"
echo "Dataset: $DATASET_PATH"
echo "LLM Model: $LLM_MODEL"
echo "Base URL: $BASE_URL"
echo "API Key: ${OPENAI_API_KEY:0:10}..." # Only show first 10 characters for security
echo "=========================================="
echo "Starting execution..."
echo ""

# Check if dataset file exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset file not found: $DATASET_PATH"
    exit 1
fi

# Execute the Python script
python run.py \
    --task "$TASK" \
    -d "$DATASET_PATH" \
    --llm "$LLM_MODEL" \
    --openai_api_key "$OPENAI_API_KEY" \
    --base_url "$BASE_URL"

# Check execution result
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Query decompose task completed successfully!"
else
    echo ""
    echo "❌ Query decompose task failed!"
    exit 1
fi

echo "=========================================="
echo "Script execution finished"
echo "=========================================="