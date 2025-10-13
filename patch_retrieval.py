import copy
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import Levenshtein
import time
import networkx as nx
from scipy.optimize import linear_sum_assignment 
import queue
from utils.utils import load_jsonl, dump_jsonl, make_needed_dir, CodexTokenizer
from utils.ccg import  create_mcpg_from_patch


class SimilarityScore:
   
    def __init__(self, gamma=0.1):
        self.gamma = gamma

    @staticmethod
    def text_edit_similarity(str1: str, str2: str):
        return 1 - Levenshtein.distance(str1, str2) / max(len(str1), len(str2))

    @staticmethod
    def text_jaccard_similarity(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return float(intersection) / union if union > 0 else 0

    @staticmethod
    def patch_content_similarity(patch1, patch2):
        content1 = ""
        content2 = ""
        
        try:
            for func in patch1.get('content', {}).get('functions', []):
                content1 += func.get('patch_function', '')
                
            for func in patch2.get('content', {}).get('functions', []):
                content2 += func.get('patch_function', '')
                
            if not content1 or not content2:
                return 0
            return SimilarityScore.text_edit_similarity(content1, content2)
        except Exception as e:
            print(f"Error computing patch content similarity: {e}")
            return 0

    @staticmethod
    def extract_diff_lines(patch_function, max_lines=30):
        if not patch_function:
            return ""

        context_lines=5 

        lines = patch_function.splitlines()
        changed_line_indices = [
            i for i, line in enumerate(lines)
            if line.strip().startswith('+') or line.strip().startswith('-')
        ]

        if not changed_line_indices:
            return "\n".join(lines[:max_lines])

        intervals = []
        for idx in changed_line_indices:
            start = max(0, idx - context_lines)
            end = min(len(lines), idx + context_lines + 1)
            intervals.append((start, end))

        merged_intervals = []
        for start, end in sorted(intervals):
            if not merged_intervals or start > merged_intervals[-1][1]:
                merged_intervals.append([start, end])
            else:
                merged_intervals[-1][1] = max(merged_intervals[-1][1], end)

        selected_lines = []
        for start, end in merged_intervals:
            selected_lines.extend(range(start, end))

        selected_lines = sorted(set(selected_lines))
        if len(selected_lines) > max_lines:
            must_keep = set(changed_line_indices)
            keep_indices = sorted(must_keep)
            remaining_slots = max_lines - len(keep_indices)

            if remaining_slots > 0:
                context_candidates = [i for i in selected_lines if i not in must_keep]
                keep_indices.extend(context_candidates[:remaining_slots])
                keep_indices = sorted(set(keep_indices))
            else:
                keep_indices = keep_indices[:max_lines]

            return "\n".join(lines[i] for i in keep_indices)

        return "\n".join(lines[i] for i in selected_lines)


    @staticmethod
    def patch_text_similarity(patch1, patch2):
        
        try:
            tokenizer = CodexTokenizer()
            patch1_diff_texts = []
            patch2_diff_texts = []
            if isinstance(patch1, str):
                diff_text = SimilarityScore.extract_diff_lines(patch1)
                if diff_text:
                    patch1_diff_texts.append(diff_text)
            elif isinstance(patch1, dict):
                content = patch1.get('content', {})
                functions = content.get('functions', [])
                if isinstance(functions, list):
                    for func in functions:
                        if isinstance(func, dict):
                            patch_func = func.get('patch_function', '')
                            if patch_func:
                                diff_text = SimilarityScore.extract_diff_lines(patch_func)
                                if diff_text:
                                    patch1_diff_texts.append(diff_text)
                elif isinstance(functions, dict):
                    patch_func = functions.get('patch_function', '')
                    if patch_func:
                        diff_text = SimilarityScore.extract_diff_lines(patch_func)
                        if diff_text:
                            patch1_diff_texts.append(diff_text)

            if isinstance(patch2, str):
                diff_text = SimilarityScore.extract_diff_lines(patch2)
                if diff_text:
                    patch2_diff_texts.append(diff_text)
            elif isinstance(patch2, dict):
                content = patch2.get('content', {})
                functions = content.get('functions', [])

                if isinstance(functions, list):
                    for func in functions:
                        if isinstance(func, dict):
                            patch_func = func.get('patch_function', '')
                            if patch_func:
                                diff_text = SimilarityScore.extract_diff_lines(patch_func)
                                if diff_text:
                                    patch2_diff_texts.append(diff_text)
                elif isinstance(functions, dict):
                    patch_func = functions.get('patch_function', '')
                    if patch_func:
                        diff_text = SimilarityScore.extract_diff_lines(patch_func)
                        if diff_text:
                            patch2_diff_texts.append(diff_text)
            patch1_text = "\n".join(patch1_diff_texts)
            patch2_text = "\n".join(patch2_diff_texts)

            if not patch1_text or not patch2_text:
                return 0
         
            patch1_tokens = tokenizer.tokenize(patch1_text)
            patch2_tokens = tokenizer.tokenize(patch2_text)

            return SimilarityScore.text_jaccard_similarity(patch1_tokens, patch2_tokens)
            
        except Exception as e:
            print(f"Error computing patch text similarity: {e}")
            return 0



    @staticmethod
    def subgraph_edit_similarity(query_graph: nx.MultiDiGraph, graph: nx.MultiDiGraph, gamma=0.1):

        if not query_graph or not query_graph.nodes or not graph or not graph.nodes:
            return 0.0           
        query_root = max(query_graph.nodes)
        root = max(graph.nodes)
        
        tokenizer = CodexTokenizer()
        
        query_root_lines = query_graph.nodes[query_root].get('sourceLines', [])
        root_lines = graph.nodes[root].get('sourceLines', [])
        
        query_graph_node_embedding = tokenizer.tokenize("".join(query_root_lines))
        graph_node_embedding = tokenizer.tokenize("".join(root_lines))
        node_sim = SimilarityScore.text_jaccard_similarity(query_graph_node_embedding, graph_node_embedding)

        node_match = dict()
        match_queue = queue.Queue()
        match_queue.put((query_root, root, 0))
        node_match[query_root] = (root, 0)

        query_graph_visited = {query_root}
        graph_visited = {root}

        graph_nodes = set(graph.nodes)

        while not match_queue.empty():
            v, u, hop = match_queue.get()
            v_neighbors = (set(query_graph.neighbors(v)) | set(query_graph.predecessors(v))) - set(query_graph_visited)
            u_neighbors = graph_nodes - set(graph_visited)

            sim_score = []
            for vn in v_neighbors:
                for un in u_neighbors:
                    vn_lines = query_graph.nodes[vn].get('sourceLines', [])
                    un_lines = graph.nodes[un].get('sourceLines', [])
                    
                    query_graph_node_embedding = tokenizer.tokenize("".join(vn_lines))
                    graph_node_embedding = tokenizer.tokenize("".join(un_lines))
                    sim = SimilarityScore.text_jaccard_similarity(query_graph_node_embedding, graph_node_embedding)
                    sim_score.append((sim, vn, un))
            sim_score.sort(key=lambda x: -x[0])
            for sim, vn, un in sim_score:
                if vn not in query_graph_visited and un not in graph_visited:
                    match_queue.put((vn, un, hop + 1))
                    node_match[vn] = (un, hop + 1)
                    query_graph_visited.add(vn)
                    graph_visited.add(un)
                    v_neighbors.remove(vn)
                    u_neighbors.remove(un)
                    node_sim += (gamma ** (hop + 1)) * sim
                if len(v_neighbors) == 0 or len(u_neighbors) == 0:
                    break
            if len(v_neighbors) != 0:
                for vn in v_neighbors:
                    node_match[vn] = None
                    query_graph_visited.add(vn)

        edge_sim = 0
        for v in query_graph.nodes:
            if v not in node_match.keys():
                node_match[v] = None
        for v_query, u_query, t in query_graph.edges:
            if node_match[v_query] is not None and node_match[u_query] is not None:
                v, hop_v = node_match[v_query]
                u, hop_u = node_match[u_query]
                if graph.has_edge(v, u, t):
                    edge_sim += (gamma ** hop_v)

        graph_sim = node_sim + edge_sim
        return graph_sim


class PatchSearchWorker:
    def __init__(self, query_patches, repository_patches, output_path, mode='coarse', max_top_k=10, gamma=0.1):
        self.query_patches = query_patches
        self.repository_patches = repository_patches
        self.output_path = output_path
        self.mode = mode
        self.max_top_k = max_top_k
        self.gamma = gamma
        
        if mode not in ['coarse', 'fine', 'coarse2fine']:
            print(f"Warning: Unknown mode '{mode}', falling back to 'coarse'")
            self.mode = 'coarse'
    
    def _compute_text_similarity(self, query_patch, repo_patch):

        sim = SimilarityScore.patch_text_similarity(query_patch, repo_patch)
        return repo_patch, sim
    
    
    def _split_patch_into_functions(self, patch_data):

        try:
            
            mcpg_result = create_mcpg_from_patch(patch_data)

            if not mcpg_result or 'function_mcpgs' not in mcpg_result:
                # print(f"Warning: Failed to create MCPG from patch {patch_data.get('patch_path', '')}.")
                return []

            function_mcpg_data = mcpg_result.get('function_mcpgs', [])
            if not function_mcpg_data:
                print(f"Warning: No function MCPGs were generated for patch {patch_data.get('patch_path', '')}.")
                return []

            function_graphs = []
            for func_mcpg in function_mcpg_data:
                graph = func_mcpg.get('mcpg')
                if isinstance(graph, nx.MultiDiGraph) and graph.nodes:
                    function_graphs.append(graph)
            
            
            return function_graphs

        except Exception as e:
            import traceback
            patch_id = patch_data.get('patch_path', 'unknown')
            print(f"Error splitting patch '{patch_id}' into functions: {e}")
            traceback.print_exc()
            return []

    def _calculate_similarity_between_graphs(self, graph1, graph2):

        is_mcpg = self._is_mcpg(graph1) and self._is_mcpg(graph2)
        
        if is_mcpg:
            return SimilarityScore.subgraph_edit_similarity(
                graph1, graph2, gamma=self.gamma
            )
        

    def _build_similarity_matrix(self, query_funcs, repo_funcs):

        num_query = len(query_funcs)
        num_repo = len(repo_funcs)
        
        sim_matrix = np.zeros((num_query, num_repo))

        for i in range(num_query):
            for j in range(num_repo):
                sim_matrix[i, j] = self._calculate_similarity_between_graphs(
                    query_funcs[i], repo_funcs[j]
                )
        
        return sim_matrix

    def _compute_aggregated_graph_similarity(self, query_patch, repo_patch):

        try:
            query_funcs = self._split_patch_into_functions(query_patch)
            repo_funcs = self._split_patch_into_functions(repo_patch)

            if not query_funcs or not repo_funcs:
                return repo_patch, 0.0

            num_query_funcs = len(query_funcs)
            num_repo_funcs = len(repo_funcs)
            
            aggregated_similarity = 0.0

            if num_query_funcs == 1 and num_repo_funcs == 1:
                aggregated_similarity = self._calculate_similarity_between_graphs(query_funcs[0], repo_funcs[0])

            elif num_query_funcs == 1 and num_repo_funcs > 1:
                sim_scores = [self._calculate_similarity_between_graphs(query_funcs[0], rf) for rf in repo_funcs]
                if sim_scores:
                    max_similarity = max(sim_scores)
                    penalty_factor = 1.0/num_repo_funcs
                    aggregated_similarity = max_similarity * penalty_factor
                else:
                    aggregated_similarity = 0.0

            elif num_query_funcs > 1 and num_repo_funcs == 1:
                sim_scores = [self._calculate_similarity_between_graphs(qf, repo_funcs[0]) for qf in query_funcs]
                if sim_scores:
                    max_similarity = max(sim_scores)
                    penalty_factor = 1.0/num_repo_funcs
                    aggregated_similarity = max_similarity * penalty_factor
                else:
                    aggregated_similarity = 0.0
            
            elif num_query_funcs > 1 and num_repo_funcs > 1:
                sim_matrix = self._build_similarity_matrix(query_funcs, repo_funcs)

                cost_matrix = 1 - sim_matrix
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                matched_similarities = sim_matrix[row_ind, col_ind]

                if matched_similarities.size > 0:
                    total_matched_similarity = matched_similarities.mean()

                    denominator = max(num_query_funcs, num_repo_funcs)
                    aggregated_similarity = total_matched_similarity / denominator
                else:
                    aggregated_similarity = 0.0 

            else:
                aggregated_similarity = 0.0

            return repo_patch, aggregated_similarity

        except Exception as e:
            import traceback
            print(f"Error computing aggregated graph similarity: {e}")
            traceback.print_exc()
            return repo_patch, 0.0
            
    def _is_mcpg(self, graph):

        if not graph or not graph.nodes:
            return False
        
        for node_id in graph.nodes:
            node_data = graph.nodes[node_id]
            if 'version' in node_data:
                return True
        return False
    
    
    

    def _find_similar_patches_coarse(self, query_patch):

        start_time = time.time()
        search_result = copy.deepcopy(query_patch)
      
        similar_patches = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            compute_sim = partial(self._compute_text_similarity, query_patch)
            futures = executor.map(compute_sim, self.repository_patches)
            similar_patches = list(futures)

        similar_patches = sorted(similar_patches, key=lambda x: x[1], reverse=True)
        top_k_patches = similar_patches[:self.max_top_k]

        search_result['similar_patches'] = [
            {
                'patch_path': p[0].get('patch_path', ''),
                'tool': p[0].get('tool', ''),
                'bug_id': p[0].get('bug_id', ''),
                'label': p[0].get('label', -1),
                'similarity': p[1],
                'content': p[0].get('content', {})
            } for p in top_k_patches
        ]
        
        end_time = time.time()
        search_result['search_runtime'] = end_time - start_time
        search_result['text_runtime'] = end_time - start_time
        search_result['graph_runtime'] = 0
        
        patch_id = query_patch.get('patch_path', 'unknown')
        print(f'Patch {patch_id} coarse search completed in {end_time - start_time:.2f}s')
        
        return search_result
    
    def _find_similar_patches_fine(self, query_patch):
        start_time = time.time()
        search_result = copy.deepcopy(query_patch)

        similar_patches_scores = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            compute_sim = partial(self._compute_aggregated_graph_similarity, query_patch)
            futures = executor.map(compute_sim, self.repository_patches)
            similar_patches_scores = list(futures)

        similar_patches_scores = sorted(similar_patches_scores, key=lambda x: x[1], reverse=True)
        top_k_patches = similar_patches_scores[:self.max_top_k]
        
        search_result['similar_patches'] = [
            {
                'patch_path': p[0].get('patch_path', ''),
                'tool': p[0].get('tool', ''),
                'bug_id': p[0].get('bug_id', ''),
                'label': p[0].get('label', -1),
                'similarity': p[1],
                'content': p[0].get('content', {})
            } for p in top_k_patches
        ]
        
        end_time = time.time()
        search_result['search_runtime'] = end_time - start_time
        search_result['text_runtime'] = 0
        search_result['graph_runtime'] = end_time - start_time
        
        patch_id = query_patch.get('patch_path', 'unknown')
        print(f'Patch {patch_id} fine search completed in {end_time - start_time:.2f}s')
        
        return search_result
    
    def _find_similar_patches_coarse2fine(self, query_patch):
        start_time = time.time()
        search_result = copy.deepcopy(query_patch)
        
        text_start_time = time.time()
        similar_patches_phase1 = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            compute_sim = partial(self._compute_text_similarity, query_patch)
            futures = executor.map(compute_sim, self.repository_patches)
            similar_patches_phase1 = list(futures)
        
        candidates_count = min(50, len(similar_patches_phase1))
        similar_patches_phase1 = sorted(similar_patches_phase1, key=lambda x: x[1], reverse=True)
        candidate_patches = [p[0] for p in similar_patches_phase1[:candidates_count]]
        text_end_time = time.time()
        
        graph_start_time = time.time()
        similar_patches_phase2_scores = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            compute_sim = partial(self._compute_aggregated_graph_similarity, query_patch)
            futures = executor.map(compute_sim, candidate_patches)
            similar_patches_phase2_scores = list(futures) 
        
        similar_patches_phase2_scores = sorted(similar_patches_phase2_scores, key=lambda x: x[1], reverse=True)
        # final_count = min(self.max_top_k, len(similar_patches_phase2_scores))
        final_count = min(self.max_top_k, len(similar_patches_phase2_scores))
        top_k_patches = similar_patches_phase2_scores[:final_count]
        graph_end_time = time.time()

        search_result['similar_patches'] = [
            {
                'patch_path': p[0].get('patch_path', ''),
                'tool': p[0].get('tool', ''),
                'bug_id': p[0].get('bug_id', ''),
                'label': p[0].get('label', -1),
                'similarity': p[1],
                'content': p[0].get('content', {})
            } for p in top_k_patches
        ]
        
        end_time = time.time()
        search_result['search_runtime'] = end_time - start_time
        search_result['text_runtime'] = text_end_time - text_start_time
        search_result['graph_runtime'] = graph_end_time - graph_start_time
        
        patch_id = query_patch.get('patch_path', 'unknown')
        print(f'Patch {patch_id} coarse2fine search completed in {end_time - start_time:.2f}s (text: {text_end_time - text_start_time:.2f}s -> {candidates_count} candidates, graph: {graph_end_time - graph_start_time:.2f}s -> {final_count} results)')
        
        return search_result
        
    def _find_similar_patches(self, query_patch):

        if self.mode == 'coarse':
            return self._find_similar_patches_coarse(query_patch)
        elif self.mode == 'fine':
            return self._find_similar_patches_fine(query_patch)
        elif self.mode == 'coarse2fine':
            return self._find_similar_patches_coarse2fine(query_patch)
        else:
            
            return self._find_similar_patches_coarse(query_patch)
        
    def run(self):

        all_start_time = time.time()
        query_patches_with_similar = []
        
        print(f"Running patch search in '{self.mode}' mode...")
        
        for query_patch in self.query_patches:
            result = self._find_similar_patches(query_patch)
            query_patches_with_similar.append(result)
            
        all_end_time = time.time()
        print(f'Total search time: {all_end_time - all_start_time:.2f}s')
        
        make_needed_dir(self.output_path)
        dump_jsonl(query_patches_with_similar, self.output_path)
        print(f'Search results saved to {self.output_path}')
        
        return query_patches_with_similar


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Retrieve similar patches for correctness evaluation')
    parser.add_argument('--query_path',  required=True, help='Path to the query patches JSONL file')
    parser.add_argument('--repository_path',  required=True, help='Path to the repository patches JSONL file')
    parser.add_argument('--output_path', required=True, help='Path to save search results')
    parser.add_argument('--mode', default='coarse', choices=['coarse', 'fine', 'coarse2fine'], 
                       help='Similarity computation mode: coarse (text-based), fine (graph-based), or coarse2fine (two-phase)')
    parser.add_argument('--max_top_k', type=int, default=10, help='Number of most similar patches to retrieve')
    parser.add_argument('--gamma', type=float, default=0.1, help='Decay factor for graph similarity calculation')
    
    args = parser.parse_args()
    
    query_patches = load_jsonl(args.query_path)
    repository_patches = load_jsonl(args.repository_path)
    
    print(f"Loaded {len(query_patches)} query patches and {len(repository_patches)} repository patches")
    
    searcher = PatchSearchWorker(
        query_patches,
        repository_patches,
        args.output_path,
        args.mode,
        args.max_top_k,
        args.gamma
    )
    searcher.run() 
