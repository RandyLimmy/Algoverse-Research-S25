#!/bin/bash
# Run all 12 presets with 10 rows, LLM judge, and save to accuracy_score_output_v4.csv
# Continues running even if individual presets fail

echo "🚀 Starting all 12 presets (10 rows each, with LLM judge)..."
echo "📊 Accuracy will be saved to: Outputs/improved_gpt5_accuracy_metrics.csv"
echo "⏰ Started at: $(date)"
echo ""

# Activate venv
source .venv/bin/activate

# Use GPT-4o for LLM judge (more accurate than gpt-3.5-turbo)
export OPENAI_JUDGE_MODEL="gpt-4o"

# Generate one random seed for all presets (so they test on the same cases)
RANDOM_SEED=$RANDOM
export PIPELINE_RANDOM_SEED=$RANDOM_SEED
echo "🎲 Random seed for this batch: $RANDOM_SEED (all presets will use same 10 cases)"
echo ""

# Track successes and failures
SUCCESS=0
FAILED=0

# Helper function to run with error handling
run_preset() {
    local name=$1
    local preset=$2
    local output_file=$3
    echo "📊 [$name/12] $preset..."
    if python3 run_ablations.py --preset "$preset" --rows 10 --use_llm_judge --judge_model gpt-4o --output_file "$output_file"; then
        echo "   ✅ Success → $output_file"
        ((SUCCESS++))
    else
        echo "   ❌ Failed (continuing...)"
        ((FAILED++))
    fi
    echo ""
}

# LLM-only variants
run_preset 1 llm_zero_shot_top5 "Outputs/improved_gpt5_llm_zero_shot_top5_n10.csv"
run_preset 2 llm_guided "Outputs/improved_gpt5_llm_guided_n10.csv"
run_preset 3 llm_guided_ontology "Outputs/improved_gpt5_llm_guided_ontology_n10.csv"

# Semantic RAG (k=3, 5, 10)
run_preset 4 semantic_rag_k3 "Outputs/improved_gpt5_semantic_rag_k3_n10.csv"
run_preset 5 semantic_rag_k5 "Outputs/improved_gpt5_semantic_rag_k5_n10.csv"
run_preset 6 semantic_rag_k10 "Outputs/improved_gpt5_semantic_rag_k10_n10.csv"

# Hybrid RAG (k=3, 5, 10)
run_preset 7 hybrid_rag_k3 "Outputs/improved_gpt5_hybrid_rag_k3_n10.csv"
run_preset 8 hybrid_rag_k5 "Outputs/improved_gpt5_hybrid_rag_k5_n10.csv"
run_preset 9 hybrid_rag_k10 "Outputs/improved_gpt5_hybrid_rag_k10_n10.csv"

# RAG variants (removed misleading "cot" from names)
run_preset 10 rag_cot "Outputs/improved_gpt5_rag_n10.csv"
run_preset 11 ontology_rag_cot "Outputs/improved_gpt5_ontology_rag_n10.csv"

# SciSpacy UMLS (gets ALL 7 improvements!)
run_preset 12 scispacy_umls "Outputs/improved_gpt5_scispacy_umls_n10.csv"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Batch run completed!"
echo "⏰ Finished at: $(date)"
echo "📊 Success: $SUCCESS/12 | Failed: $FAILED/12"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 Results:"
echo "   - Individual predictions: Outputs/improved_gpt5_*_n10.csv"
echo "   - Detailed logs: Outputs/improved_gpt5_*_n10_detailed_log.csv"
echo "   - Accuracy summary: Outputs/improved_gpt5_accuracy_metrics.csv"
echo ""
echo "🎯 Compare improved scispacy_umls against baseline:"
echo "   Baseline: Outputs/gpt5_scispacy_umls_no_cot_n100.csv (35% soft, 100 rows)"
echo "   Improved: Outputs/improved_gpt5_scispacy_umls_n10.csv (check improved_gpt5_accuracy_metrics.csv, 10 rows)"


