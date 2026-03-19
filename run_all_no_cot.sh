#!/bin/bash
# Run all experiments with NO CoT prompting
# All results will be saved with "_no_cot" suffix and metrics auto-saved to gpt5_metrics_summary_v2.csv

set -e  # Exit on error

# Activate virtual environment
source .venv/bin/activate

echo "🚀 Starting all experiments (NO CoT prompting)..."
echo "All accuracy metrics will be automatically saved to Outputs/gpt5_metrics_summary_v2.csv"
echo ""

# LLM-only runs
echo "📊 Running LLM-only experiments..."
python3 run_ablations.py --preset llm_zero_shot_top5 --rows 100 --output_file Outputs/gpt5_llm_zero_shot_top5_no_cot_n100.csv --use_llm_judge
python3 run_ablations.py --preset llm_guided --rows 100 --output_file Outputs/gpt5_llm_guided_no_cot_n100.csv --use_llm_judge
python3 run_ablations.py --preset llm_guided_ontology --rows 100 --output_file Outputs/gpt5_llm_guided_ontology_no_cot_n100.csv --use_llm_judge

# Semantic RAG runs
echo ""
echo "📊 Running Semantic RAG experiments..."
python3 run_ablations.py --preset semantic_rag_k3 --rows 100 --output_file Outputs/gpt5_semantic_rag_k3_no_cot_n100.csv --use_llm_judge
python3 run_ablations.py --preset semantic_rag_k5 --rows 100 --output_file Outputs/gpt5_semantic_rag_k5_no_cot_n100.csv --use_llm_judge
python3 run_ablations.py --preset semantic_rag_k10 --rows 100 --output_file Outputs/gpt5_semantic_rag_k10_no_cot_n100.csv --use_llm_judge

# Hybrid RAG runs
echo ""
echo "📊 Running Hybrid RAG experiments..."
python3 run_ablations.py --preset hybrid_rag_k3 --rows 100 --output_file Outputs/gpt5_hybrid_rag_k3_no_cot_n100.csv --use_llm_judge
python3 run_ablations.py --preset hybrid_rag_k5 --rows 100 --output_file Outputs/gpt5_hybrid_rag_k5_no_cot_n100.csv --use_llm_judge
python3 run_ablations.py --preset hybrid_rag_k10 --rows 100 --output_file Outputs/gpt5_hybrid_rag_k10_no_cot_n100.csv --use_llm_judge

# RAG CoT runs
echo ""
echo "📊 Running RAG CoT experiments..."
python3 run_ablations.py --preset rag_cot --rows 100 --output_file Outputs/gpt5_rag_no_cot_n100.csv --use_llm_judge
python3 run_ablations.py --preset ontology_rag_cot --rows 100 --output_file Outputs/gpt5_ontology_rag_no_cot_n100.csv --use_llm_judge

# SciSpacy UMLS run
echo ""
echo "📊 Running SciSpacy UMLS experiment..."
python3 run_ablations.py --preset scispacy_umls --rows 100 --output_file Outputs/gpt5_scispacy_umls_no_cot_n100.csv --use_llm_judge

echo ""
echo "✅ All experiments completed!"
echo "📋 Check results:"
echo "   - Individual predictions: Outputs/gpt5_*_no_cot_n100.csv"
echo "   - Consolidated metrics: Outputs/gpt5_metrics_summary_v2.csv"
