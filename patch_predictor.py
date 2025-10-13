import copy
import argparse
import os
import time
from tqdm import tqdm
import openai
import torch
from utils.utils import load_jsonl, dump_jsonl, make_needed_dir, CodexTokenizerN, StarCoderTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from patch_retrieval import SimilarityScore


class PatchCorrectnessPredictor:

    def __init__(self, model='gpt-3.5-turbo-instruct', max_context_tokens=None, temperature=0, api_key="", top_k=10, models_cache_dir="./models_cache"):

        self.model_name = model
        self.max_context_tokens = max_context_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_cache_dir = models_cache_dir
        
        if model.startswith('gpt-'):
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
            else:
                self.openai_client = openai.OpenAI()
            self.tokenizer = CodexTokenizer()
            self.max_context_tokens = max_context_tokens or (4096 - 16)
            self.model = None
            self.model_type = "openai"
        elif model.startswith('starcoder2-'):
            # # Starcoder2 family (e.g. starcoder2-7b, starcoder2-15b)
            model_path = f"./models_cache/bigcode/{self.model_name}/"
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    use_safetensors=True,
                    torch_dtype=torch.float16,
                    device_map = "auto",
                    # low_cpu_mem_usage=True,
                )
                tokenizer_raw = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                if getattr(tokenizer_raw, "pad_token", None) is None and getattr(tokenizer_raw, "eos_token",
                                                                                 None) is not None:
                    tokenizer_raw.pad_token = tokenizer_raw.eos_token


                self.tokenizer = StarCoderTokenizer(tokenizer_raw)
                self.raw_tokenizer = tokenizer_raw
                self.max_context_tokens = 16384 - 16
                self.model_type = "huggingface"
                self.generation_config = GenerationConfig(
                    max_new_tokens=16,
                    do_sample=False,
                    #temperature=0,
                    eos_token_id=tokenizer_raw.eos_token_id,
                    pad_token_id=tokenizer_raw.pad_token_id,
                )
            except Exception as e:
                raise ValueError(f"Error loading Starcoder model: {str(e)}")
        elif model.startswith('codellama-'):
            try:
                model_path = os.path.join(models_cache_dir, model)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    #low_cpu_mem_usage=True,
                    device_map="auto",
                )
                self.raw_tokenizer = AutoTokenizer.from_pretrained(model_path)                
                if self.raw_tokenizer.pad_token is None and self.raw_tokenizer.eos_token_id is not None:
                    self.raw_tokenizer.pad_token = self.raw_tokenizer.eos_token
                self.tokenizer = self.raw_tokenizer
                self.max_context_tokens = 16384 - 16
                self.model_type = "huggingface"
                self.generation_config = GenerationConfig(
                    max_new_tokens=16,
                    do_sample= False,
                    # temperature=temperature if temperature > 0 else 0,
                    eos_token_id=self.raw_tokenizer.eos_token_id,
                    pad_token_id=self.raw_tokenizer.pad_token_id,
                )
            except Exception as e:
                raise ValueError(f"Error loading CodeLlama model: {str(e)}")
        elif model.startswith('codegen2-'):
            try:
                model_path = os.path.join(models_cache_dir, model)
                tokenizer_path = os.path.join(models_cache_dir, f"{model}-tokenizer")
                
                self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
                tokenizer_raw = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
                from utils.utils import CodeGenTokenizer
                self.tokenizer = CodeGenTokenizer(tokenizer_raw)
                self.raw_tokenizer = tokenizer_raw
                self.max_context_tokens = max_context_tokens or (2048 - 16)
                self.model_type = "huggingface"
                self.generation_config = GenerationConfig(
                    max_new_tokens= 16,
                    do_sample=False,
                    #temperature=temperature if temperature > 0 else 0,
                    eos_token_id=tokenizer_raw.eos_token_id,
                    pad_token_id=tokenizer_raw.pad_token_id,
                )
            except Exception as e:
                raise ValueError(f"Error loading CodeGen2 model: {str(e)}")
        else:
            raise ValueError(f"Unsupported model: {model}")
            
        print(f"Initialized predictor with model: {model} on device: {self.device}")
        if self.model_type == "huggingface":
            print(f"Model loaded successfully: {self.model.__class__.__name__}")

    
    def _build_patch_prompt(self, patch):

        #prompt = f"Tool: {patch.get('tool', 'Unknown')}\n"
        prompt = f"Bug ID: {patch.get('bug_id', 'Unknown')}\n"
        
        content = patch.get('content', {})
        functions = content.get('functions', [])
        if isinstance(functions, dict):
            functions = [functions]
        elif not isinstance(functions, list):
            functions = []

        if functions:
            for i, func in enumerate(functions):
                if not isinstance(func, dict):
                    prompt += f"\nFunction {i+1}/{len(functions)}: [Invalid function data, skipped]\n"
                    continue
                prompt += f"\nFunction {i+1}/{len(functions)}:\n"
                prompt += f"File: {func.get('path', 'unknown')}\n"
                prompt += f"Lines: {func.get('start_loc', '?')}-{func.get('end_loc', '?')}\n"
                prompt += "```diff\n"
                full_patch_function = func.get('patch_function', '')                
                concise_diff = SimilarityScore.extract_diff_lines(full_patch_function, max_lines=50) 
                prompt += concise_diff
                prompt += "```\n"
        else:
            prompt += "\nNo function information available.\n"
            
        return prompt
    
    def _build_evaluation_prompt(self, query_patch, similar_patches):
        
        base_prompt = "\n# You are expert code reviewer. Let's check patch correctness. 'WRONG' means although it can pass test cases but cannot fix the bug. 'CORRECT' means it really addresses the bug.\n"


        # add Bug description 
        bug_description = query_patch.get('bug_description')
        if bug_description:
            base_prompt += "\n## The bug refers to:\n"
            if isinstance(bug_description, list):
                base_prompt += "\n".join(bug_description) + "\n"
            else:
                base_prompt += str(bug_description) + "\n"

        # add execution trace 
        execution_traces = query_patch.get('execution_traces')
        if execution_traces:
            base_prompt += "\n## The execution traces of the bug are: \n"
            trace_lines = []
            for trace in execution_traces:  
                trace_lines.extend(trace.split('\n'))
            base_prompt += '\n'.join(trace_lines[:30])

         # add test cases
        test_cases = query_patch.get('test_cases')
        if test_cases and isinstance(test_cases, dict):
            base_prompt += "\n## Originally the buggy code cannot pass some failing test cases and now the patchedd code can pass them. Those failing test cases are: \n"
            for test_name, test_code in test_cases.items():
                base_prompt += f"{test_name.strip()}\n"
                # base_prompt += "```java\n"
                if test_code is not None:
                    base_prompt += test_code.strip() + "\n"
                base_prompt += "\n"
                             
        # add code coverage summary
        coverage_summary = query_patch.get('coverage_summary')
        if coverage_summary:
            base_prompt += "\n## Although this patch can pass available test cases, the available test cases only cover limited coverages: \n"
            base_prompt += coverage_summary.strip() + "\n"

        if self.top_k > 0 and similar_patches and len(similar_patches) > self.top_k:
            similar_patches = similar_patches[:self.top_k]

        if similar_patches:
            base_prompt += "\n## For your reference, there are labeled examples to the query patch:"
            for i, patch in enumerate(similar_patches):
                similar_patch_prompt = f"\n--- Similar Patch {i + 1} ---\n"
                similar_patch_prompt += self._build_patch_prompt(patch)
                label = patch.get('label', -1)
                if label != -1:
                    similar_patch_prompt += f"\nLabel: {'WRONG' if label == 1 else 'CORRECT'}\n"

                similarity = patch.get('similarity', 0.0)
                similar_patch_prompt += f"Similarity: {similarity:.4f}\n"            
                current_prompt_str = base_prompt + similar_patch_prompt

                if len(self.raw_tokenizer.encode(current_prompt_str)) > self.max_context_tokens - 4000:
                    
                    print(f"Warning: Prompt limit reached. Stopped adding similar patches at index {i}.")
                    break
                base_prompt = base_prompt + similar_patch_prompt
        
       
        final_prompt = '\n This is query patch.'
        final_prompt += self._build_patch_prompt(query_patch)
        final_prompt += '\n Q: It was wrong or correct? A: It was '
        
        prompt = base_prompt + final_prompt
        #print(f"Prompt length: {len(self.tokenizer.encode(prompt))}")
        
        
        
        return prompt
    
    
    def _call_model(self, prompt):
        if self.model_type == "openai":
            completion = self.openai_client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_context_tokens=self.max_context_tokens,
                temperature=self.temperature
            )
            return completion.choices[0].text.strip()
        
        elif self.model_type == "huggingface":
            if not self.model:
                raise ValueError("Model not initialized")
                
            if self.model_name.startswith("starcoder2") or self.model_name.startswith("codegen2-"):
                prompt_ids = self.raw_tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    response_ids = self.model.generate(
                        prompt_ids['input_ids'],
                        generation_config=self.generation_config,
                        attention_mask=prompt_ids['attention_mask']
                    )
                
                full_response = self.raw_tokenizer.decode(response_ids[0], skip_special_tokens=True)
                prompt_lines = prompt.splitlines(keepends=True)
                n_prompt_lines = len(prompt_lines)
                response_lines = full_response.splitlines(keepends=True)
                response = "".join(response_lines[n_prompt_lines:])
                return response.strip()
                
            elif self.model_name.startswith("codellama-"):
                inputs = self.raw_tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        generation_config=self.generation_config      
                    )
                
                full_response = self.raw_tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = full_response[len(prompt):]
                return response.strip()
        
        raise ValueError(f"Unknown model type: {self.model_type}")
    def predict_single_patch(self, query_patch):
        patch_id = query_patch.get('patch_path', 'unknown')
        similar_patches = query_patch.get('similar_patches', [])
        prompt = self._build_evaluation_prompt(query_patch, similar_patches)
        
        try:
            raw_output = self._call_model(prompt)            
            prediction = 0 if "CORRECT" in raw_output.upper() else 1       
            true_label = query_patch.get('label', -1)
            
            return patch_id, prediction, raw_output, true_label
            
        except Exception as e:
            print(f"Error predicting patch {patch_id}: {e}")
            return patch_id, -1, f"ERROR: {str(e)}", query_patch.get('label', -1)
    
    def predict_patches(self, query_patches_with_similar):

        results = []
        
        for query_patch in tqdm(query_patches_with_similar, desc="Predicting patches"):
            patch_id, prediction, raw_output, true_label = self.predict_single_patch(query_patch)
            
            result = copy.deepcopy(query_patch)
            result['prediction'] = prediction
            result['raw_prediction_output'] = raw_output
            result['true_label'] = true_label
            result['used_model'] = self.model_name
            result['used_top_k'] = self.top_k if self.top_k > 0 else len(query_patch.get('similar_patches', []))
            
            results.append(result)
            
        return results


def main(args):
    input_data = load_jsonl(args.input_path)
    print(f"Loaded {len(input_data)} patches with similar results")

    predictor = PatchCorrectnessPredictor(
        model=args.model,
        max_context_tokens=args.max_context_tokens,
        temperature=args.temperature,
        api_key=args.api_key,
        top_k=args.top_k,
        models_cache_dir=args.models_cache_dir
    )

    results = predictor.predict_patches(input_data)
    
    make_needed_dir(args.output_path)
    dump_jsonl(results, args.output_path)
    print(f"Results saved to {args.output_path}")

    predictions = [r.get('prediction', -1) for r in results]
    true_labels = [r.get('true_label', -1) for r in results]
    
    correct_predictions = sum(1 for p, t in zip(predictions, true_labels) 
                             if p == t and t != -1)
    valid_samples = sum(1 for t in true_labels if t != -1)
    
    if valid_samples > 0:
        accuracy = correct_predictions / valid_samples
        print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{valid_samples})")
    else:
        print("No valid samples with labels for evaluation")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict patch correctness using LLM')
    parser.add_argument('--input_path', required=True, help='Path to the retrieved patches JSONL file')
    parser.add_argument('--output_path', required=True, help='Path to save prediction results')
    parser.add_argument('--model', default='codellama-7b-hf', 
                        choices=['gpt-4-turbo', 'codegen2-7b', 'codellama-7b-hf', 'codellama-13b',
                                 'starcoder2-3b', 'starcoder2-7b', 'starcoder2-15b'],
                        help='LLM model to use for prediction')
    parser.add_argument('--max_context_tokens', type=int, default=2048, help='Maximum tokens for generation')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for sampling (0 for deterministic)')
    parser.add_argument('--api_key', default="", help='OpenAI API key')
    parser.add_argument('--top_k', type=int, default=10, help='Number of similar patches to use for prediction (-1 for all)')
    parser.add_argument('--models_cache_dir', default='./models_cache', help='Directory for cached models')
    
    args = parser.parse_args()

    main(args) 
