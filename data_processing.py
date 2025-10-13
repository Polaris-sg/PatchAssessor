import json
#import copy
import os
from utils.utils import load_jsonl, dump_jsonl, make_needed_dir


class PatchDataProcessor:
    
    
    tool_name = ['ACS', 'Arja','AVATAR', 'CapGen', 'Cardumen', 'DynaMoth', 'FixMiner',
        'GenProg','HDRepair',  'Jaid', 'jGenProg', 'jKali', 'jMutRepair',  'Kali', 'kPAR',
        'Nopol', 'RSRepair', 'SequenceR', 'SimFix', 'SketchFix', 'SOFix', 'TBar' ]  
    
    def __init__(self, data_path=None, tool_name=None):
        self.data_path = data_path
        self.tool_name = tool_name
        self.data = self._load_data()
        
    def _load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def split_query_and_repository(self):
        results = {}
        
        for tool in PatchDataProcessor.tool_name:
            query_set = []
            repository_set = []
            
            for patch_path, patch_info in self.data.items():
                patch_record = {
                    'patch_path': patch_path,
                    'tool': patch_info.get('tool', ''),
                    'bug_id': patch_info.get('bug_id', ''),
                    'label': patch_info.get('label', -1), 
                    'content': patch_info.get('content', {}),
                    'test_cases': patch_info.get('test_cases', {}),
                    'coverage_summary': patch_info.get('coverage_summary', ''),
                    'bug_description': patch_info.get('bug_description', ''),
                    'execution_traces': patch_info.get('execution_traces', '')
                }
                
                if patch_info.get('tool') == tool:
                    query_set.append(patch_record)
                else:
                    repository_set.append(patch_record)
            
            results[tool] = {
                'query_set': query_set,
                'repository_set': repository_set
            }
            
            print(f"Tool '{tool}': Query set size = {len(query_set)}, Repository set size = {len(repository_set)}")
        
        return results
    
    
    def save_processed_data(self, output_dir=None):

        if output_dir is None:
            output_dir = './processed_data/'
        
        make_needed_dir(output_dir)
        split_results = self.split_query_and_repository()        
        file_paths = {}
        
        for tool, data_sets in split_results.items():
            query_path = os.path.join(str(output_dir), f"{tool}_query_set.jsonl")
            dump_jsonl(data_sets['query_set'], query_path)
  
            repo_path = os.path.join(str(output_dir), f"{tool}_repository_set.jsonl")
            dump_jsonl(data_sets['repository_set'], repo_path)
            
            file_paths[tool] = {
                'query_set': query_path,
                'repository_set': repo_path
            }
        
        return file_paths
        


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process patch data for correctness evaluation')
    parser.add_argument('--data_path', default='PCA_Func_dataset.json', help='Path to the JSON data file')
    parser.add_argument('--output_dir', default='./processed_data/', help='Output directory for processed data')
    
    args = parser.parse_args()

    processor = PatchDataProcessor(args.data_path)
    file_paths = processor.save_processed_data(args.output_dir)
    
    print(f"Data processed successfully for all tools")
    print(f"Files saved to: {args.output_dir}")
    for tool, paths in file_paths.items():
        print(f"Tool '{tool}':")
        for key, path in paths.items():
            print(f"  {key}: {path}")
