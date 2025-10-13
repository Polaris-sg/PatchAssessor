#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import time

from patch_retrieval import PatchSearchWorker
from patch_predictor import PatchCorrectnessPredictor
from evaluation import PatchCorrectnesEvaluator
from utils.utils import load_jsonl, make_needed_dir, dump_jsonl

def parse_args():
    parser = argparse.ArgumentParser(description="Patch Correctness Evaluation Pipeline")
    
    # 指定已处理数据和结果的存储目录
    parser.add_argument('--processed_data_dir', default='./processed_data', help='Directory containing the processed datasets.')
    parser.add_argument('--output_dir', default='./results', help='Output directory for all results')
    
    # 检索参数
    parser.add_argument('--retrieval_mode', default='coarse', choices=['coarse', 'fine', 'coarse2fine'], 
                      help='Patch similarity computation mode: coarse (text-based), fine (graph-based), or coarse2fine (two-phase)')
    parser.add_argument('--max_top_k', type=int, default=10, help='Number of most similar patches to retrieve')
    parser.add_argument('--gamma', type=float, default=0.1, help='Decay factor for graph similarity computation')
    
    # 预测参数
    parser.add_argument('--model', default='codellama-7b-hf', 
                       choices=['gpt-3.5-turbo-instruct', 'gpt-4-turbo', 'codellama-7b-hf', 'codellama-13b', 'starcoder2-3b', 'starcoder2-7b', 'starcoder2-15b'],
                       help='LLM model to use for prediction')
    parser.add_argument('--top_k', type=int, default=10, help='Number of similar patches to use for prediction (-1 for all)')
    parser.add_argument('--models_cache_dir', default='./models_cache', help='Directory for cached models')
    
    # 为了兼容PatchCorrectnessPredictor的调用，保留以下参数
    parser.add_argument('--max_context_tokens', type=int, default=2048, help='Maximum number of context tokens for the model')
    # parser.add_argument('--temperature', type=float, default=0.2, help='Temperature for sampling (0 for deterministic)')
    # parser.add_argument('--api_key', default="", help='OpenAI API key')
    
    # 控制流程参数
    parser.add_argument('--skip_retrieval', action='store_true', help='Skip retrieval step')
    parser.add_argument('--skip_prediction', action='store_true', help='Skip prediction step')
    parser.add_argument('--skip_evaluation', action='store_true', help='Skip evaluation step')
    
    return parser.parse_args()


def run_pipeline(args):
    total_start_time = time.time()
    
    # 定义所有子数据集的 tool_name
    tool_names = [
        'ACS', 'Arja', 'AVATAR', 'CapGen', 'Cardumen', 'DynaMoth',
        'FixMiner', 'GenProg', 'HDRepair', 'Jaid', 'jGenProg',
        'jKali', 'jMutRepair', 'Kali', 'kPAR', 'Nopol', 'RSRepair',
        'SequenceR', 'SimFix', 'SketchFix', 'SOFix', 'TBar'
    ]
    
    # 根据模型名称创建用于文件名的简称
    llm_name = args.model.replace('-hf', '').replace('-instruct', '').replace('.', '')

    mode_name = args.retrieval_mode
    # 创建顶层输出目录
    make_needed_dir(args.output_dir)
    search_results_dir = os.path.join(args.output_dir, 'search_results')
    prediction_results_dir = os.path.join(args.output_dir, 'prediction_results')
    evaluation_results_dir = os.path.join(args.output_dir, 'evaluation_results')
    make_needed_dir(search_results_dir)
    make_needed_dir(prediction_results_dir)
    make_needed_dir(evaluation_results_dir)

    # 遍历所有 tool_name，执行流水线
    for tool_name in tool_names:
        print(f"\n{'='*25} Processing tool: {tool_name} {'='*25}")
        tool_start_time = time.time()

        # 根据 tool_name 和 llm_name 动态生成文件路径
        query_path = os.path.join(args.processed_data_dir, f"{tool_name}_query_set.jsonl")
        repository_path = os.path.join(args.processed_data_dir, f"{tool_name}_repository_set.jsonl")
        
        search_output_path = os.path.join(search_results_dir, f"{tool_name}_{llm_name}_{mode_name}.jsonl")
        prediction_output_path = os.path.join(prediction_results_dir, f"{tool_name}_{llm_name}_{mode_name}.jsonl")
        metrics_output_path = os.path.join(evaluation_results_dir, f"{tool_name}_{llm_name}_{mode_name}.json")

        # 步骤 1: 相似补丁检索
        if not args.skip_retrieval:
            print(f"\n=== Step 1: Similar Patch Retrieval for {tool_name} ===")
            retrieval_start_time = time.time()
            
            if not os.path.exists(query_path) or not os.path.exists(repository_path):
                print(f"Error: Input files for {tool_name} not found. Skipping.")
                print(f"  - Missing Query File: {query_path}")
                print(f"  - Missing Repository File: {repository_path}")
                continue  # 跳过当前工具，处理下一个

            query_patches = load_jsonl(query_path)
            repository_patches = load_jsonl(repository_path)
            print(f"Loaded {len(query_patches)} query patches and {len(repository_patches)} repository patches.")
            
            searcher = PatchSearchWorker(
                query_patches=query_patches,
                repository_patches=repository_patches,
                output_path=search_output_path,
                mode=args.retrieval_mode,
                max_top_k=args.max_top_k,
                gamma=args.gamma
            )
            searcher.run()
            print(f"Retrieval results saved to {search_output_path}")
            retrieval_end_time = time.time()
            print(f"Retrieval for {tool_name} completed in {retrieval_end_time - retrieval_start_time:.2f}s")

        # 步骤 2: 补丁正确性预测
        if not args.skip_prediction:
            print(f"\n=== Step 2: Patch Correctness Prediction for {tool_name} ===")
            prediction_start_time = time.time()
            
            if not os.path.exists(search_output_path):
                print(f"Error: Search results not found for {tool_name}. Please run retrieval first. Skipping.")
                print(f"  - Missing File: {search_output_path}")
                continue

            search_results = load_jsonl(search_output_path)
            print(f"Loaded {len(search_results)} patches with search results.")
            
            predictor = PatchCorrectnessPredictor(
                model=args.model,
                max_context_tokens=args.max_context_tokens,
                # temperature=args.temperature,
                api_key=args.api_key,
                top_k=args.top_k,
                models_cache_dir=args.models_cache_dir
            )
            
            prediction_results = predictor.predict_patches(search_results)
            
            dump_jsonl(prediction_results, prediction_output_path)
            print(f"Prediction results saved to {prediction_output_path}")
            prediction_end_time = time.time()
            print(f"Prediction for {tool_name} completed in {prediction_end_time - prediction_start_time:.2f}s")

        # 步骤 3: 评估
        if not args.skip_evaluation:
            print(f"\n=== Step 3: Evaluation for {tool_name} ===")
            evaluation_start_time = time.time()
            
            if not os.path.exists(prediction_output_path):
                print(f"Error: Prediction results not found for {tool_name}. Please run prediction first. Skipping.")
                print(f"  - Missing File: {prediction_output_path}")
                continue

            evaluator = PatchCorrectnesEvaluator(prediction_output_path)
            metrics = evaluator.compute_metrics()
            
            print(f"Metrics for {tool_name}:")
            evaluator.print_metrics(metrics)
            
           
            evaluator.save_metrics(metrics_output_path)
            print(f"Evaluation metrics saved to {metrics_output_path}")
            evaluation_end_time = time.time()
            print(f"Evaluation for {tool_name} completed in {evaluation_end_time - evaluation_start_time:.2f}s")
        
        tool_end_time = time.time()
        print(f"\nTotal time for {tool_name}: {tool_end_time - tool_start_time:.2f}s")

    total_end_time = time.time()
    print(f"\nTotal pipeline execution time: {total_end_time - total_start_time:.2f}s")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)