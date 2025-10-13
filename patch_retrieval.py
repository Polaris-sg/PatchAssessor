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
        """计算两个字符串的编辑距离相似度"""
        return 1 - Levenshtein.distance(str1, str2) / max(len(str1), len(str2))

    @staticmethod
    def text_jaccard_similarity(list1, list2):
        """计算两个列表的Jaccard相似度"""
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return float(intersection) / union if union > 0 else 0

    @staticmethod
    def patch_content_similarity(patch1, patch2):
        """计算两个补丁内容的相似度 - 文本相似度（coarse模式）"""
        # 提取补丁内容 - 所有修改的函数内容拼接
        content1 = ""
        content2 = ""
        
        try:
            for func in patch1.get('content', {}).get('functions', []):
                content1 += func.get('patch_function', '')
                
            for func in patch2.get('content', {}).get('functions', []):
                content2 += func.get('patch_function', '')
                
            # 如果没有内容可比较，返回0
            if not content1 or not content2:
                return 0
                
            # 使用编辑距离计算相似度
            return SimilarityScore.text_edit_similarity(content1, content2)
        except Exception as e:
            print(f"Error computing patch content similarity: {e}")
            return 0

    @staticmethod
    def extract_diff_lines(patch_function, max_lines=30):
        if not patch_function:
            return ""

        # 每个改动行前后保留的上下文行数
        context_lines=5 

        lines = patch_function.splitlines()

        # 找出所有修改行索引
        changed_line_indices = [
            i for i, line in enumerate(lines)
            if line.strip().startswith('+') or line.strip().startswith('-')
        ]

        if not changed_line_indices:
            return "\n".join(lines[:max_lines])

        # 生成改动区间（包含上下文）
        intervals = []
        for idx in changed_line_indices:
            start = max(0, idx - context_lines)
            end = min(len(lines), idx + context_lines + 1)
            intervals.append((start, end))

        # 合并重叠区间
        merged_intervals = []
        for start, end in sorted(intervals):
            if not merged_intervals or start > merged_intervals[-1][1]:
                merged_intervals.append([start, end])
            else:
                merged_intervals[-1][1] = max(merged_intervals[-1][1], end)

        # 收集所有需要的行
        selected_lines = []
        for start, end in merged_intervals:
            selected_lines.extend(range(start, end))

        # 去重并排序
        selected_lines = sorted(set(selected_lines))

        # 如果超过 max_lines，优先保留改动行，再补上下文
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
            
            # 提取各自补丁函数中的差异内容
            patch1_diff_texts = []
            patch2_diff_texts = []

            # 处理patch1
            if isinstance(patch1, str):
                # 如果是字符串，直接提取差异
                diff_text = SimilarityScore.extract_diff_lines(patch1)
                if diff_text:
                    patch1_diff_texts.append(diff_text)
            elif isinstance(patch1, dict):
                # 处理字典格式的patch
                content = patch1.get('content', {})
                functions = content.get('functions', [])

                # 检查functions是否为列表或单个字典
                if isinstance(functions, list):
                    for func in functions:
                        if isinstance(func, dict):
                            patch_func = func.get('patch_function', '')
                            if patch_func:
                                diff_text = SimilarityScore.extract_diff_lines(patch_func)
                                if diff_text:
                                    patch1_diff_texts.append(diff_text)
                elif isinstance(functions, dict):
                    # 单个函数的情况
                    patch_func = functions.get('patch_function', '')
                    if patch_func:
                        diff_text = SimilarityScore.extract_diff_lines(patch_func)
                        if diff_text:
                            patch1_diff_texts.append(diff_text)

            # 处理patch2（同样的逻辑）
            if isinstance(patch2, str):
                # 如果是字符串，直接提取差异
                diff_text = SimilarityScore.extract_diff_lines(patch2)
                if diff_text:
                    patch2_diff_texts.append(diff_text)
            elif isinstance(patch2, dict):
                # 处理字典格式的patch
                content = patch2.get('content', {})
                functions = content.get('functions', [])

                # 检查functions是否为列表或单个字典
                if isinstance(functions, list):
                    for func in functions:
                        if isinstance(func, dict):
                            patch_func = func.get('patch_function', '')
                            if patch_func:
                                diff_text = SimilarityScore.extract_diff_lines(patch_func)
                                if diff_text:
                                    patch2_diff_texts.append(diff_text)
                elif isinstance(functions, dict):
                    # 单个函数的情况
                    patch_func = functions.get('patch_function', '')
                    if patch_func:
                        diff_text = SimilarityScore.extract_diff_lines(patch_func)
                        if diff_text:
                            patch2_diff_texts.append(diff_text)

            # 合并所有差异文本
            patch1_text = "\n".join(patch1_diff_texts)
            patch2_text = "\n".join(patch2_diff_texts)
            
            # 如果任一补丁没有提取到差异内容，返回0
            if not patch1_text or not patch2_text:
                return 0
            
            # 将文本转换为token序列
            patch1_tokens = tokenizer.tokenize(patch1_text)
            patch2_tokens = tokenizer.tokenize(patch2_text)
            
            # 使用Jaccard相似度计算
            return SimilarityScore.text_jaccard_similarity(patch1_tokens, patch2_tokens)
            
        except Exception as e:
            print(f"Error computing patch text similarity: {e}")
            return 0



    @staticmethod
    def subgraph_edit_similarity(query_graph: nx.MultiDiGraph, graph: nx.MultiDiGraph, gamma=0.1):
        """
        计算两个图的编辑相似度（基于原有的tree-sitter实现）
        """
        if not query_graph or not query_graph.nodes or not graph or not graph.nodes:
            return 0.0
            
        # 获取根节点（使用最大ID作为根节点）
        query_root = max(query_graph.nodes)
        root = max(graph.nodes)
        
        tokenizer = CodexTokenizer()
        
        # 计算根节点相似度
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
    """
    补丁检索工作器，用于从补丁库中检索与查询补丁相似的补丁
    """
    def __init__(self, query_patches, repository_patches, output_path, mode='coarse', max_top_k=10, gamma=0.1):
        """
        初始化补丁检索工作器
        
        Args:
            query_patches (list): 查询补丁列表
            repository_patches (list): 补丁库补丁列表
            output_path (str): 检索结果保存路径
            mode (str): 检索模式：
                - 'coarse': 仅文本级别相似度比较
                - 'fine': 仅图级别相似度比较
                - 'coarse2fine': 两阶段检索（先文本检索50个候选，再图检索10个最相似）
            max_top_k (int): 返回的最相似补丁数量
            gamma (float): 图相似度计算中的衰减系数
        """
        self.query_patches = query_patches
        self.repository_patches = repository_patches
        self.output_path = output_path
        self.mode = mode
        self.max_top_k = max_top_k
        self.gamma = gamma
        
        # 验证模式参数
        if mode not in ['coarse', 'fine', 'coarse2fine']:
            print(f"Warning: Unknown mode '{mode}', falling back to 'coarse'")
            self.mode = 'coarse'
    
    def _compute_text_similarity(self, query_patch, repo_patch):
        """
        计算查询补丁和库中补丁的文本相似度
        
        Args:
            query_patch: 查询补丁
            repo_patch: 库中补丁
            
        Returns:
            tuple: (repo_patch, 相似度得分)
        """
        # 使用文本相似度计算
        sim = SimilarityScore.patch_text_similarity(query_patch, repo_patch)
        return repo_patch, sim
    
    
    def _split_patch_into_functions(self, patch_data):
        """
        将单个补丁拆分为一个函数级MCPG（融合代码属性图）的列表。

        Args:
            patch_data (dict): 包含补丁信息的字典。

        Returns:
            list[nx.MultiDiGraph]: 代表每个函数MCPG的图对象列表。
                                  如果无法创建或没有有效函数，则返回空列表。
        """
        try:
            # 1. 调用外部工具构建包含所有函数图的结果
            mcpg_result = create_mcpg_from_patch(patch_data)

            if not mcpg_result or 'function_mcpgs' not in mcpg_result:
                # print(f"Warning: Failed to create MCPG from patch {patch_data.get('patch_path', '')}.")
                return []

            function_mcpg_data = mcpg_result.get('function_mcpgs', [])
            if not function_mcpg_data:
                print(f"Warning: No function MCPGs were generated for patch {patch_data.get('patch_path', '')}.")
                return []

            # 2. 提取每个函数有效的MCPG图
            function_graphs = []
            for func_mcpg in function_mcpg_data:
                graph = func_mcpg.get('mcpg')
                # 确保图不为空
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
        """
        计算两个函数图之间的相似度分数（可替换的模块）。

        Args:
            graph1 (nx.MultiDiGraph): 第一个图。
            graph2 (nx.MultiDiGraph): 第二个图。

        Returns:
            float: 0到1之间的相似度分数。
        """
        # 修正了原始代码中的逻辑：is_mcpg应调用mcpg_edit_similarity
        is_mcpg = self._is_mcpg(graph1) and self._is_mcpg(graph2)
        
        if is_mcpg:
            return SimilarityScore.subgraph_edit_similarity(
                graph1, graph2, gamma=self.gamma
            )
        

    def _build_similarity_matrix(self, query_funcs, repo_funcs):
        """
        构建查询函数和仓库函数之间的相似度矩阵。

        Args:
            query_funcs (list[nx.MultiDiGraph]): 查询补丁的函数图列表。
            repo_funcs (list[nx.MultiDiGraph]): 仓库补丁的函数图列表。

        Returns:
            np.ndarray: 一个 M x N 的矩阵，其中 M 是查询函数数，N 是仓库函数数。
                        matrix[i, j] 是 query_funcs[i] 和 repo_funcs[j] 的相似度。
        """
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
        """
        计算两个补丁之间聚合后的图相似度。
        该函数合并了图的生成和基于函数粒度的相似度聚合逻辑。

        Args:
            query_patch (dict): 查询补丁。
            repo_patch (dict): 仓库中的补丁。

        Returns:
            tuple[dict, float]: (仓库补丁, 聚合后的相似度分数)。
        """
        try:
            # 1. 将查询补丁和仓库补丁都拆分为函数级图的列表
            query_funcs = self._split_patch_into_functions(query_patch)
            repo_funcs = self._split_patch_into_functions(repo_patch)

            # 如果任一补丁无法被拆分为函数图，则认为它们不相似
            if not query_funcs or not repo_funcs:
                return repo_patch, 0.0

            num_query_funcs = len(query_funcs)
            num_repo_funcs = len(repo_funcs)
            
            aggregated_similarity = 0.0

            # 2. 根据函数数量应用不同的聚合策略
            # 案例 1: 单函数 vs 单函数
            if num_query_funcs == 1 and num_repo_funcs == 1:
                aggregated_similarity = self._calculate_similarity_between_graphs(query_funcs[0], repo_funcs[0])

            # 案例 2: 单函数 vs 多函数
            elif num_query_funcs == 1 and num_repo_funcs > 1:
                sim_scores = [self._calculate_similarity_between_graphs(query_funcs[0], rf) for rf in repo_funcs]
                if sim_scores:
                    max_similarity = max(sim_scores)
                    penalty_factor = 1.0/num_repo_funcs
                    aggregated_similarity = max_similarity * penalty_factor
                else:
                    aggregated_similarity = 0.0

            # 案例 3: 多函数 vs 单函数
            elif num_query_funcs > 1 and num_repo_funcs == 1:
                sim_scores = [self._calculate_similarity_between_graphs(qf, repo_funcs[0]) for qf in query_funcs]
                if sim_scores:
                    max_similarity = max(sim_scores)
                    penalty_factor = 1.0/num_repo_funcs
                    aggregated_similarity = max_similarity * penalty_factor
                else:
                    aggregated_similarity = 0.0
            
            # 案例 4: 多函数 vs 多函数 (使用匈牙利算法)
            elif num_query_funcs > 1 and num_repo_funcs > 1:
                # a. 构建相似度矩阵
                sim_matrix = self._build_similarity_matrix(query_funcs, repo_funcs)
                
                # b. 使用匈牙利算法找到最优匹配
                # 注意: linear_sum_assignment 解决的是成本最小化问题，
                # 因此我们需要将相似度矩阵转换为成本矩阵 (cost = 1 - similarity)。
                cost_matrix = 1 - sim_matrix
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # c. 提取最优匹配对的相似度
                matched_similarities = sim_matrix[row_ind, col_ind]
                
                # d. 计算匹配对的平均相似度作为最终得分
                if matched_similarities.size > 0:
                    total_matched_similarity = matched_similarities.mean()
                    # 使用两个补丁中较大的函数数量作为分母。
                    # 这相当于 (匹配对平均分 * 结构一致性分数)
                    # 其中 结构一致性分数 = 匹配数量 / max(M, N)
                    denominator = max(num_query_funcs, num_repo_funcs)
                    aggregated_similarity = total_matched_similarity / denominator
                else:
                    aggregated_similarity = 0.0 # 如果没有匹配项
            
            # 其他情况（例如某个列表为空），默认相似度为 0
            else:
                aggregated_similarity = 0.0

            return repo_patch, aggregated_similarity

        except Exception as e:
            import traceback
            print(f"Error computing aggregated graph similarity: {e}")
            traceback.print_exc()
            return repo_patch, 0.0
            
    def _is_mcpg(self, graph):
        """
        检查图是否为MCPG（通过检查节点是否有version属性）
        """
        if not graph or not graph.nodes:
            return False
        
        # 检查是否有节点包含version属性
        for node_id in graph.nodes:
            node_data = graph.nodes[node_id]
            if 'version' in node_data:
                return True
        return False
    
    
    

    def _find_similar_patches_coarse(self, query_patch):
        """
        使用文本相似度检索相似补丁（coarse模式）
        
        Args:
            query_patch: 查询补丁
            
        Returns:
            dict: 原查询补丁加上检索到的相似补丁
        """
        start_time = time.time()
        search_result = copy.deepcopy(query_patch)
        
        # 使用线程池并行计算相似度
        similar_patches = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            compute_sim = partial(self._compute_text_similarity, query_patch)
            futures = executor.map(compute_sim, self.repository_patches)
            similar_patches = list(futures)
        
        # 按相似度排序并取top-k
        similar_patches = sorted(similar_patches, key=lambda x: x[1], reverse=True)
        top_k_patches = similar_patches[:self.max_top_k]
        
        # 添加相似补丁到结果
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
        """
        使用图相似度检索相似补丁（fine模式）
        
        Args:
            query_patch: 查询补丁
            
        Returns:
            dict: 原查询补丁加上检索到的相似补丁
        """
        start_time = time.time()
        search_result = copy.deepcopy(query_patch)
        
        # 使用线程池并行计算相似度
        similar_patches_scores = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            compute_sim = partial(self._compute_aggregated_graph_similarity, query_patch)
            futures = executor.map(compute_sim, self.repository_patches)
            # futures现在返回(repo_patch, similarity)元组，直接使用
            similar_patches_scores = list(futures)

        # 按相似度排序并取top-k
        similar_patches_scores = sorted(similar_patches_scores, key=lambda x: x[1], reverse=True)
        top_k_patches = similar_patches_scores[:self.max_top_k]
        
        # 添加相似补丁到结果
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
        """
        两阶段检索相似补丁（coarse2fine模式）
        
        Args:
            query_patch: 查询补丁
            
        Returns:
            dict: 原查询补丁加上检索到的相似补丁
        """
        start_time = time.time()
        search_result = copy.deepcopy(query_patch)
        
        # 第一阶段：文本相似度（coarse）筛选
        text_start_time = time.time()
        similar_patches_phase1 = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            compute_sim = partial(self._compute_text_similarity, query_patch)
            futures = executor.map(compute_sim, self.repository_patches)
            similar_patches_phase1 = list(futures)
        
        # 按相似度排序并取前50个作为候选集（固定候选数量）
        candidates_count = min(30, len(similar_patches_phase1))
        #candidates_count = min(50, len(similar_patches_phase1))
        similar_patches_phase1 = sorted(similar_patches_phase1, key=lambda x: x[1], reverse=True)
        candidate_patches = [p[0] for p in similar_patches_phase1[:candidates_count]]
        text_end_time = time.time()
        
        # 第二阶段：图相似度（fine）精排
        graph_start_time = time.time()
        similar_patches_phase2_scores = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            compute_sim = partial(self._compute_aggregated_graph_similarity, query_patch)
            futures = executor.map(compute_sim, candidate_patches)
            # futures现在返回(repo_patch, similarity)元组，直接使用
            similar_patches_phase2_scores = list(futures) 
        
        # 按图相似度排序并取top-10（固定返回10个最相似补丁）
        similar_patches_phase2_scores = sorted(similar_patches_phase2_scores, key=lambda x: x[1], reverse=True)
        # final_count = min(self.max_top_k, len(similar_patches_phase2_scores))
        final_count = min(self.max_top_k, len(similar_patches_phase2_scores))
        top_k_patches = similar_patches_phase2_scores[:final_count]
        graph_end_time = time.time()
        
        # 添加相似补丁到结果
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
        """
        为单个查询补丁找到最相似的补丁，根据mode选择检索策略
        
        Args:
            query_patch: 查询补丁
            
        Returns:
            dict: 原查询补丁加上检索到的相似补丁
        """
        if self.mode == 'coarse':
            return self._find_similar_patches_coarse(query_patch)
        elif self.mode == 'fine':
            return self._find_similar_patches_fine(query_patch)
        elif self.mode == 'coarse2fine':
            return self._find_similar_patches_coarse2fine(query_patch)
        else:
            # 默认使用coarse模式
            return self._find_similar_patches_coarse(query_patch)
        
    def run(self):
        """
        运行补丁检索，为所有查询补丁找到相似补丁
        
        Returns:
            list: 带有检索结果的查询补丁列表
        """
        all_start_time = time.time()
        query_patches_with_similar = []
        
        print(f"Running patch search in '{self.mode}' mode...")
        
        for query_patch in self.query_patches:
            result = self._find_similar_patches(query_patch)
            query_patches_with_similar.append(result)
            
        all_end_time = time.time()
        print(f'Total search time: {all_end_time - all_start_time:.2f}s')
        
        # 保存结果
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
    
    # 加载数据
    query_patches = load_jsonl(args.query_path)
    repository_patches = load_jsonl(args.repository_path)
    
    print(f"Loaded {len(query_patches)} query patches and {len(repository_patches)} repository patches")
    
    # 进行检索
    searcher = PatchSearchWorker(
        query_patches,
        repository_patches,
        args.output_path,
        args.mode,
        args.max_top_k,
        args.gamma
    )
    searcher.run() 