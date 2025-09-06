#!/bin/bash

# QA Task Selection Script
# Description: Choose between direct QA or EoG QA tasks

# =============================================================================
# Configuration Variables
# =============================================================================

# Direct QA Configuration
DIRECT_QA_TASK="direct_qa"
DIRECT_QA_DATASET="preprocess_datasets/feature_enrich_datasets/cwq_gpt-4o-mini-2024-07-18_feature_enrich_2025-09-06_19-35-38.parquet"

# EoG QA Configuration
EOG_QA_TASK="eog_qa"
EOG_QA_DATASET="preprocess_datasets/feature_enrich_datasets/cwq_gpt-4o-mini-2024-07-18_feature_enrich_2025-09-06_19-35-38.parquet"

# Common LLM Configuration
LLM_MODEL="gpt-4o-mini-2024-07-18"
OPENAI_API_KEY="sk-SgiEuM72oCrNUpDZ9b87F351103e4d218d69B42e36C859Df"
BASE_URL="https://api.key77qiqi.cn/v1"

# =============================================================================
# Helper Functions
# =============================================================================

show_menu() {
    echo "=========================================="
    echo "QA Task Selection Menu"
    echo "=========================================="
    echo "Please choose which QA task to run:"
    echo ""
    echo "1) Direct QA"
    echo "   Dataset: $DIRECT_QA_DATASET"
    echo ""
    echo "2) EoG QA"
    echo "   Dataset: $EOG_QA_DATASET"
    echo ""
    echo "3) Exit"
    echo "=========================================="
    echo "Common Configuration:"
    echo "LLM Model: $LLM_MODEL"
    echo "Base URL: $BASE_URL"
    echo "API Key: ${OPENAI_API_KEY:0:10}..."
    echo "=========================================="
}

run_direct_qa() {
    echo ""
    echo "🔄 Running Direct QA"
    echo "=========================================="
    echo "Task: $DIRECT_QA_TASK"
    echo "Dataset: $DIRECT_QA_DATASET"
    echo "=========================================="
    
    # Check if dataset exists
    if [ ! -f "$DIRECT_QA_DATASET" ]; then
        echo "❌ Error: Direct QA dataset not found: $DIRECT_QA_DATASET"
        return 1
    fi
    
    echo "Starting direct QA..."
    python run.py \
        --task "$DIRECT_QA_TASK" \
        -d "$DIRECT_QA_DATASET" \
        --llm "$LLM_MODEL" \
        --openai_api_key "$OPENAI_API_KEY" \
        --base_url "$BASE_URL"
    
    if [ $? -eq 0 ]; then
        echo "✅ Direct QA completed successfully!"
        return 0
    else
        echo "❌ Direct QA failed!"
        return 1
    fi
}

run_eog_qa() {
    echo ""
    echo "🔄 Running EoG QA"
    echo "=========================================="
    echo "Task: $EOG_QA_TASK"
    echo "Dataset: $EOG_QA_DATASET"
    echo "=========================================="
    
    # Check if dataset exists
    if [ ! -f "$EOG_QA_DATASET" ]; then
        echo "❌ Error: EoG QA dataset not found: $EOG_QA_DATASET"
        return 1
    fi
    
    echo "Starting EoG QA..."
    python run.py \
        --task "$EOG_QA_TASK" \
        -d "$EOG_QA_DATASET" \
        --llm "$LLM_MODEL" \
        --openai_api_key "$OPENAI_API_KEY" \
        --base_url "$BASE_URL"
    
    if [ $? -eq 0 ]; then
        echo "✅ EoG QA completed successfully!"
        return 0
    else
        echo "❌ EoG QA failed!"
        return 1
    fi
}

# =============================================================================
# Main Script Execution
# =============================================================================

# Check if command line argument is provided
if [ $# -eq 1 ]; then
    case $1 in
        1|direct)
            run_direct_qa
            exit $?
            ;;
        2|eog)
            run_eog_qa
            exit $?
            ;;
        *)
            echo "❌ Invalid argument. Use: 1/direct or 2/eog"
            exit 1
            ;;
    esac
fi

# Interactive mode
while true; do
    show_menu
    echo -n "Enter your choice [1-3]: "
    read choice
    
    case $choice in
        1)
            run_direct_qa
            echo ""
            echo "Press Enter to continue..."
            read
            ;;
        2)
            run_eog_qa
            echo ""
            echo "Press Enter to continue..."
            read
            ;;
        3)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo ""
            echo "❌ Invalid option. Please choose 1-3."
            echo "Press Enter to continue..."
            read
            ;;
    esac
done