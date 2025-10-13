from turtle import right
import networkx as nx
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language
from utils.utils import extract_function_from_patch
import os


JAVA_LANGUAGE = get_language('java')

def _extract_identifiers(node, src_lines, line_offset=0):
    
    identifiers = set()
    
   
    def find_ids(n):       
        if n.type == 'identifier':
            start_row, start_col = n.start_point
            end_row, end_col = n.end_point
            original_start_row = start_row - line_offset
            original_end_row = end_row - line_offset
            if original_start_row == original_end_row and original_start_row >= 0 and original_start_row < len(src_lines):
                # .strip() to remove potential leading/trailing whitespace
                identifier_name = src_lines[original_start_row][start_col:end_col].strip()
                if identifier_name: 
                    identifiers.add(identifier_name)
        for child in n.children:
            find_ids(child)

    find_ids(node)
    return identifiers

def java_control_dependence_graph(root_node, CCG, src_lines, parent, line_offset=0):   
        
    node_id = len(CCG.nodes)
    new_parent = parent
    
    target_node_types = [
        'import_declaration', 'class_declaration', 'method_declaration', 'enum_declaration', 'interface_declaration', 'while_statement', 'for_statement',
         'if_statement', 'try_statement', 
        'else_clause', 'catch_clause', 'finally_clause', 
        'expression_statement', 'local_variable_declaration', 'return_statement',
        'break_statement', 'continue_statement', 'ERROR'
    ]
    
    is_graph_node = root_node.type in target_node_types or \
                    'statement' in root_node.type or \
                    'declaration' in root_node.type

    if is_graph_node:
        start_row = root_node.start_point[0] - line_offset   
        end_row = root_node.end_point[0] - line_offset      
        
        start_row = max(0, start_row)
        end_row = max(0, end_row)
        
        source_lines = src_lines[start_row:end_row + 1] if start_row < len(src_lines) else []
        CCG.add_node(node_id, nodeType=root_node.type,
                     startRow=start_row, endRow=end_row,
                     sourceLines=source_lines,
                     defSet=set(), 
                     useSet=set())
        
        if parent is not None:
            CCG.add_edge(parent, node_id, 'CDG')
        
        new_parent = node_id
        # print(f"Added node {node_id}: type={root_node.type}, startRow={start_row}, endRow={end_row}, source={source_lines}")

    for child in root_node.children:
        java_control_dependence_graph(child, CCG, src_lines, new_parent, line_offset)

    if new_parent is not None:
        if root_node.type == 'variable_declarator':
            name_node = root_node.child_by_field_name('name')
            if name_node and name_node.type == 'identifier':
                start_row, start_col = name_node.start_point
                end_row, end_col = name_node.end_point
                if start_row == end_row and start_row < len(src_lines):
                    var_name = src_lines[start_row][start_col:end_col].strip()
                    CCG.nodes[new_parent]['defSet'].add(var_name)
                    #print(f"Node {new_parent}: Added '{var_name}' to defSet (declaration)")
        elif root_node.type == 'assignment_expression':
            left_node = root_node.child_by_field_name('left')

            if left_node and left_node.type == 'identifier':
                start_row, start_col = left_node.start_point
                end_row, end_col = left_node.end_point
                if start_row == end_row and start_row < len(src_lines):
                    var_name = src_lines[start_row][start_col:end_col].strip()
                    CCG.nodes[new_parent]['defSet'].add(var_name)
                    #print(f"Node {new_parent}: Added '{var_name}' to defSet (assignment)")

            elif left_node and left_node.type == 'field_access':

                 rightmost_field = None                 
                 for child in left_node.children:
                     if child.type == 'field_identifier':
                      start_row, start_col = child.start_point
                      end_row, end_col = child.end_point
                      if start_row == end_row and start_row < len(src_lines):
                          rightmost_field  = src_lines[start_row][start_col:end_col].strip()
                          break
                    
                 if rightmost_field:
                     CCG.nodes[new_parent]['defSet'].add(rightmost_field)
                     #print(f"Node {new_parent}: Added '{rightmost_field}' to defSet (field assignment)")                 
        elif root_node.type == 'formal_parameter':
            name_node = root_node.child_by_field_name('name')
            if name_node and name_node.type == 'identifier':
                start_row, start_col = name_node.start_point
                end_row, end_col = name_node.end_point
                if start_row == end_row and start_row < len(src_lines):
                    var_name = src_lines[start_row][start_col:end_col].strip()
                    CCG.nodes[new_parent]['defSet'].add(var_name)
                    #print(f"Node {new_parent}: Added '{var_name}' to defSet (parameter)")
        if root_node.type == 'enhanced_for_statement':
            for child in root_node.children:
                if child.type == 'identifier':
                     # The first identifier after '(' is the variable
                     start_row, start_col = child.start_point
                     end_row, end_col = child.end_point
                     if start_row == end_row and start_row < len(src_lines):
                         var_name = src_lines[start_row][start_col:end_col].strip()
                         type_node = root_node.child_by_field_name('type')
                         if type_node and child.start_byte > type_node.end_byte:
                            CCG.nodes[new_parent]['defSet'].add(var_name)
                            #print(f"Node {new_parent}: Added '{var_name}' to defSet (for-each var)")
                            break # Assume only one is declared here



        use_node = None
        if root_node.type == 'variable_declarator':
            use_node = root_node.child_by_field_name('value')
        elif root_node.type == 'assignment_expression':
            use_node = root_node.child_by_field_name('right')

        elif root_node.type in ['if_statement', 'while_statement']:
            use_node = root_node.child_by_field_name('condition')
        elif root_node.type == 'enhanced_for_statement':
            # The collection is the last identifier in the parenthesis
            # Let's just grab all identifiers in the header and remove the defined one
            all_ids_in_header = set()
            for child in root_node.children:
                if child.type == '(':
                    continue
                if child.type == ')':
                    break
                all_ids_in_header.update(_extract_identifiers(child, src_lines))
            
            defined_vars = CCG.nodes[new_parent].get('defSet', set())
            used_vars = all_ids_in_header - defined_vars
            for var in used_vars:
                CCG.nodes[new_parent]['useSet'].add(var)
                #print(f"Node {new_parent}: Added '{var}' to useSet (for-each collection)")


        elif root_node.type == 'return_statement':
            use_node = root_node
        elif root_node.type == 'method_invocation':
            use_node = root_node
        
        if use_node:
            used_vars = _extract_identifiers(use_node, src_lines)
            if root_node.type == 'assignment_expression':
                defined_vars = CCG.nodes[new_parent].get('defSet', set())
                used_vars -= defined_vars

            for var in used_vars:
                CCG.nodes[new_parent]['useSet'].add(var)
                #print(f"Node {new_parent}: Added '{var}' to useSet (general)")

    return



def java_control_flow_graph(CCG):
    
    if not CCG or not CCG.nodes:
        #print("!!! DEBUG: Empty CCG provided")
        return nx.MultiDiGraph(), []
    
    CFG = nx.MultiDiGraph()
    next_sibling = {}
    first_children = {}
    start_nodes = []
    
    for v in CCG.nodes:
        if len(list(CCG.predecessors(v))) == 0:
            start_nodes.append(v)
    
    start_nodes.sort()
    
    if not start_nodes:
        print("!!! DEBUG: java_control_flow_graph found NO start nodes.")
        return nx.MultiDiGraph(), []
    for i in range(len(start_nodes) - 1):
        v = start_nodes[i]
        u = start_nodes[i + 1]
        next_sibling[v] = u
    next_sibling[start_nodes[-1]] = None
    
    for v in CCG.nodes:
        children = list(CCG.neighbors(v))
        if children:
            children.sort()
            
            for i in range(len(children) - 1):
                u = children[i]
                w = children[i + 1]
                if (CCG.nodes[v]['nodeType'] == 'if_statement' and 
                    'clause' in CCG.nodes[w]['nodeType']):
                    next_sibling[u] = None
                else:
                    next_sibling[u] = w
            
            next_sibling[children[-1]] = None
            first_children[v] = children[0]
        else:
            first_children[v] = None
    
    edge_list = []
    added_edges = set()  
    
    def add_edge_if_not_exists(from_node, to_node, edge_type='CFG'):
        edge_key = (from_node, to_node, edge_type)
        if edge_key not in added_edges:
            edge_list.append((from_node, to_node, edge_type))
            added_edges.add(edge_key)
    
    for v in CCG.nodes:
        node_type = CCG.nodes[v]['nodeType']
        if node_type in ['return_statement', 'break_statement', 'continue_statement']:
            continue
            
        elif node_type in ['for_statement', 'while_statement']:
            first_child = first_children.get(v)
            if first_child is not None:
                add_edge_if_not_exists(v, first_child)
            
            next_sib = next_sibling.get(v)
            if next_sib is not None:
                add_edge_if_not_exists(v, next_sib)
                
        elif node_type == 'if_statement':
            for child in CCG.neighbors(v):
                child_type = CCG.nodes[child]['nodeType']
                if child_type in ['if_statement', 'else_clause'] or 'clause' in child_type:
                    add_edge_if_not_exists(v, child)
            
            first_child = first_children.get(v)
            if first_child is not None:
                add_edge_if_not_exists(v, first_child)
                
        elif node_type == 'try_statement':
            for child in CCG.neighbors(v):
                child_type = CCG.nodes[child]['nodeType']
                if child_type in ['try_statement', 'catch_clause', 'finally_clause']:
                    add_edge_if_not_exists(v, child)
            
            first_child = first_children.get(v)
            if first_child is not None:
                add_edge_if_not_exists(v, first_child)
                
        elif 'clause' in node_type:
            first_child = first_children.get(v)
            if first_child is not None:
                add_edge_if_not_exists(v, first_child)
                
        else:
            first_child = first_children.get(v)
            if first_child is not None:
                add_edge_if_not_exists(v, first_child)
        next_sib = next_sibling.get(v)
        if next_sib is not None:
            add_edge_if_not_exists(v, next_sib)
    
    CFG.add_edges_from(edge_list)
    
    for v in CCG.nodes:
        if v not in CFG.nodes:
            CFG.add_node(v)
        
        if v in CFG.nodes:
            CFG.nodes[v].update(CCG.nodes[v])
    
    # print(f"CFG constructed with {len(CFG.nodes)} nodes and {len(CFG.edges)} edges")
    
    return CFG, edge_list


def java_data_dependence_graph(CFG, CCG):
     
    ddg_edges_added = 0
    for v in CFG.nodes():
        for u in CFG.nodes():
            if v == u or 'import' in CCG.nodes[v]['nodeType']:
                continue
            def_set = CCG.nodes[v].get('defSet', set())
            use_set = CCG.nodes[u].get('useSet', set())
            common_vars = def_set & use_set
            if not common_vars:
                continue
            # print(f"Potential DDG edge: {v} -> {u}, shared vars: {common_vars}")
            if nx.has_path(CFG, v, u):
                # print(f"  CFG path exists from {v} to {u}")
                for var in common_vars:
                    has_clean_path = False
                    for path in nx.all_shortest_paths(CFG, v, u):
                        # print(f"  Checking path: {path}")
                        redefined = False
                        for node in path[1:-1]:
                            if var in CCG.nodes[node].get('defSet', set()):
                                redefined = True
                                break
                        if not redefined:
                            has_clean_path = True
                            break
                    if has_clean_path:
                        CCG.add_edge(v, u, key='DDG', variable=var)
                        ddg_edges_added += 1
                        # print(f"  Added DDG edge: {v} -> {u} for variable {var}")
            # else:
                # print(f"  No CFG path from {v} to {u}")
    
    # print(f"Total DDG edges added: {ddg_edges_added}")
    return CCG



def create_graph(code_lines):
    
    if isinstance(code_lines, str):
        code_lines = code_lines.splitlines()
    elif isinstance(code_lines, list) and len(code_lines) == 1 and '\n' in code_lines[0]:
        code_lines = code_lines[0].splitlines()
    
    line_offset = 0
    is_function_level = True
    
    if code_lines:
        first_line = code_lines[0].strip()
        if first_line.startswith('+') or first_line.startswith('-'):
            first_line = first_line[1:].strip()
        if (first_line.startswith('package') or 
            first_line.startswith('import')):
            is_function_level = False
    
    if is_function_level:
        wrapped_lines = ['class DummyWrapper {'] + code_lines + ['}']
        code_lines = wrapped_lines
        line_offset = 1 
    
    src_lines = [line + '\n' for line in code_lines ]  
    
    parser = Parser()
    JAVA_LANGUAGE = None

    try:
        JAVA_LANGUAGE = get_language('java')
        if JAVA_LANGUAGE is not None:
            parser.set_language(JAVA_LANGUAGE)
            # print("Using tree_sitter_languages to load Java parser")
        else:
            raise Exception("Failed to load Java language from tree_sitter_languages")
    except Exception as e:
        print(f"tree_sitter_languages failed: {e}")
        try:
            language_path = './my-languages.dll' if os.name == 'nt' else './my-languages.so'
            Language.build_library(language_path, ['./tree-sitter-java'])
            JAVA_LANGUAGE = Language(language_path, 'java')
            parser.set_language(JAVA_LANGUAGE)
            # print("Using fallback tree_sitter language loading method")
        except Exception as e:
            print(f"Fallback method failed: {e}")
            return None
    
    if len(src_lines) == 0:
        print("src_lines is empty")
        return None
    
    comment_lines = []
    in_multiline_comment = False
    for i, line in enumerate(src_lines):
        #original_line = line
        line = line.lstrip()

        if in_multiline_comment:
            if '*/' in line:
                before, after = line.split('*/', 1)
                src_lines[i] = after if after.strip() else '\n'
                in_multiline_comment = False
            else:
                comment_lines.append(i)
                src_lines[i] = '\n'
        else:
            if '//' in line:
                code, _ = line.split('//', 1)
                if code.strip():  
                    src_lines[i] = code.rstrip() + '\n'
                else:  
                    src_lines[i] = '\n'
                comment_lines.append(i)

            elif '/*' in line:
                before, after = line.split('/*', 1)
                if '*/' in after:
                    _, after_comment = after.split('*/', 1)
                    src_lines[i] = before + after_comment
                else:
                    src_lines[i] = before if before.strip() else '\n'
                    in_multiline_comment = True
                comment_lines.append(i)
    #print("comment_lines:", comment_lines)
    
    def read_callable(byte_offset, point):
        row, column = point
        if row >= len(src_lines) or column >= len(src_lines[row]):
            return None
        return src_lines[row][column:].encode('utf8', errors='ignore')

    tree = parser.parse(read_callable)
    
    root_nodes = []
    if is_function_level and tree.root_node.children:
        # 寻找class_declaration -> class_body
        for child in tree.root_node.children:
            if child.type == 'class_declaration':
                for subchild in child.children:
                    if subchild.type == 'class_body':
                        root_nodes = list(subchild.children)
                        break
                break
    else:
        root_nodes = list(tree.root_node.children)
    all_comment = True
    for child in root_nodes:
        if child.type != 'comment':
            all_comment = False
    if all_comment:
        return None

    ccg = nx.MultiDiGraph()
   
    for child in root_nodes:
         java_control_dependence_graph(child, ccg, code_lines, None, line_offset)

    cfg, cfg_edge_list = java_control_flow_graph(ccg)
   

    java_data_dependence_graph(cfg, ccg)

    
    ccg.add_edges_from(cfg_edge_list)

    node_list = list(ccg.nodes)
    node_list.sort()
    comment_lines.reverse()
    max_comment_line = 0   

    for comment_line_num in comment_lines:
        insert_id = -1
        for v in ccg.nodes:
            if ccg.nodes[v]['startRow'] > comment_line_num:
                insert_id = v
                break
        if insert_id == -1:
            max_comment_line = max(max_comment_line, comment_line_num)
        else:
            ccg.nodes[insert_id]['startRow'] = comment_line_num
            end_row = ccg.nodes[insert_id]['endRow']
            ccg.nodes[insert_id]['sourceLines'] = code_lines[comment_line_num:end_row + 1]
    if max_comment_line != 0 and node_list:  # 添加node_list非空检查
        last_node_id = node_list[-1]
        ccg.nodes[last_node_id]['endRow'] = max_comment_line
        start_row = ccg.nodes[last_node_id]['startRow']
        ccg.nodes[last_node_id]['sourceLines'] = code_lines[start_row:max_comment_line + 1]
    #print("Edges in CCG:", list(ccg.edges(data=True, keys=True)))
    return ccg



def extract_patch_subgraph(old_ccg, new_ccg):
    if old_ccg is None:
        old_ccg = nx.MultiDiGraph()
    if new_ccg is None:
        new_ccg = nx.MultiDiGraph()
    
    def create_node_signature(ccg, node_id):
        if node_id not in ccg.nodes:
            return None
        node_data = ccg.nodes[node_id]
        return (
            node_data.get('nodeType', ''),
            node_data.get('startRow', -1),
            node_data.get('endRow', -1),
            tuple(sorted(node_data.get('defSet', set()))),
            tuple(sorted(node_data.get('useSet', set())))
        )
    
    old_signatures = {node_id: create_node_signature(old_ccg, node_id) 
                     for node_id in old_ccg.nodes}
    new_signatures = {node_id: create_node_signature(new_ccg, node_id) 
                     for node_id in new_ccg.nodes}
    
    old_sig_to_node = {sig: node_id for node_id, sig in old_signatures.items() if sig}
    new_sig_to_node = {sig: node_id for node_id, sig in new_signatures.items() if sig}
    
    old_sigs = set(old_sig_to_node.keys())
    new_sigs = set(new_sig_to_node.keys())
    added_sigs = new_sigs - old_sigs
    removed_sigs = old_sigs - new_sigs
    
    added_nodes = [new_sig_to_node[sig] for sig in added_sigs]
    removed_nodes = [old_sig_to_node[sig] for sig in removed_sigs]    
    subgraph = nx.MultiDiGraph()    
    relevant_nodes = set()
    
    for node_id in added_nodes:
        if node_id in new_ccg.nodes:
            relevant_nodes.add(('new', node_id))
    
    for node_id in removed_nodes:
        if node_id in old_ccg.nodes:
            relevant_nodes.add(('old', node_id))
    
    for version, node_id in list(relevant_nodes):
        ccg = new_ccg if version == 'new' else old_ccg
        for neighbor in list(ccg.predecessors(node_id)) + list(ccg.successors(node_id)):
            relevant_nodes.add((version, neighbor))
    node_mapping = {}
    subgraph_node_id = 0
    
    for version, orig_node_id in relevant_nodes:
        ccg = new_ccg if version == 'new' else old_ccg
        if orig_node_id in ccg.nodes:
            node_data = ccg.nodes[orig_node_id].copy()
            node_data['version'] = version
            node_data['original_id'] = orig_node_id
            subgraph.add_node(subgraph_node_id, **node_data)
            node_mapping[(version, orig_node_id)] = subgraph_node_id
            subgraph_node_id += 1
    
    for version, orig_node_id in relevant_nodes:
        ccg = new_ccg if version == 'new' else old_ccg
        if orig_node_id in ccg.nodes:
            for successor in ccg.successors(orig_node_id):
                if (version, successor) in node_mapping:
                    u = node_mapping[(version, orig_node_id)]
                    v = node_mapping[(version, successor)]
                    # 添加所有边（包括多重边）
                    for key, edge_data in ccg[orig_node_id][successor].items():
                        subgraph.add_edge(u, v, **edge_data)
    
    return {
        'added_nodes': added_nodes,
        'removed_nodes': removed_nodes,
        'subgraph': subgraph,
        'node_mapping': node_mapping
    }


def normalize_subgraph(subgraph):
    """规范化子图，将变量名替换为通用标识符"""
    normalized_graph = subgraph.copy()
    
    all_variables = set()
    for node_id in normalized_graph.nodes:
        node_data = normalized_graph.nodes[node_id]
        all_variables.update(node_data.get('defSet', set()))
        all_variables.update(node_data.get('useSet', set()))
    
    var_mapping = {}
    var_counter = 1
    for var in sorted(all_variables):
        if var and var.isidentifier():
            var_mapping[var] = f'var{var_counter}'
            var_counter += 1
    
    for node_id in normalized_graph.nodes:
        node_data = normalized_graph.nodes[node_id]

        if 'defSet' in node_data:
            new_def_set = set()
            for var in node_data['defSet']:
                new_def_set.add(var_mapping.get(var, var))
            normalized_graph.nodes[node_id]['defSet'] = new_def_set
        
        if 'useSet' in node_data:
            new_use_set = set()
            for var in node_data['useSet']:
                new_use_set.add(var_mapping.get(var, var))
            normalized_graph.nodes[node_id]['useSet'] = new_use_set

        if 'sourceLines' in node_data:
            normalized_lines = []
            for line in node_data['sourceLines']:
                line = ' '.join(line.split())
                if line:
                    normalized_lines.append(line)
            normalized_graph.nodes[node_id]['sourceLines'] = normalized_lines
    
    return normalized_graph


def create_mcpg_from_patch(patch_data): 

    try:
        content = patch_data.get('content', {})
        function_num = content.get('function_num', 1)

        patch_path = patch_data.get('patch_path', 'Unknown path')
        
        buggy_code_list, fixed_code_list, _ = extract_enhanced_buggy_fixed_code(patch_data)


        function_mcpgs = []

        min_length = min(len(buggy_code_list), len(fixed_code_list))
        if len(buggy_code_list) != len(fixed_code_list):
            print(f"Warning: Length mismatch - buggy_code_list: {len(buggy_code_list)}, fixed_code_list: {len(fixed_code_list)}")
            print(f"Processing first {min_length} functions only")
        
        for i in range(min_length):
            buggy_code = buggy_code_list[i] if i < len(buggy_code_list) else []
            fixed_code = fixed_code_list[i] if i < len(fixed_code_list) else []

            enhanced_cpg_result = create_enhanced_cpg_with_versions(buggy_code, fixed_code, patch_path=patch_path)
            
            buggy_ccg = enhanced_cpg_result.get('buggy_ccg')
            fixed_ccg = enhanced_cpg_result.get('fixed_ccg')

            if (buggy_ccg is None or not buggy_ccg.nodes()) and (fixed_ccg is None or not fixed_ccg.nodes()):
                print(f"Warning: Both buggy and fixed CPGs are empty, skipping this function.  Patch path:{patch_path}")
                  
                continue

            mcpg = merge_cpgs_to_mcpg(buggy_ccg, fixed_ccg)
            
            if mcpg and mcpg.nodes():
                mcpg = slice_and_prune_mcpg(mcpg)

            function_mcpgs.append({
                'original_buggy_code': buggy_code,
                'original_fixed_code': fixed_code,
                'buggy_cpg_nodes': len(buggy_ccg.nodes) if buggy_ccg else 0,
                'fixed_cpg_nodes': len(fixed_ccg.nodes) if fixed_ccg else 0,
                'mcpg_nodes': len(mcpg.nodes) if mcpg else 0,
                'mcpg_edges': len(mcpg.edges) if mcpg else 0,
                'mcpg': mcpg
            })

        return {
            'function_num': function_num,
            'function_mcpgs': function_mcpgs,
            'patch_data': patch_data
        }

    except Exception as e:
        print(f"Error in create_mcpg_from_patch: {e}")
        return {
            'function_num': 0,
            'function_mcpgs': [],
            'patch_data': patch_data,
            'error': str(e)
        }



def extract_enhanced_buggy_fixed_code(patch_data):

    try:
        
        if not isinstance(patch_data, dict) or not patch_data:
            raise ValueError("patch_data must be a non-empty dictionary")
        
        content = patch_data.get('content', {})
        function_num = content.get('function_num', 1)        
        functions = content.get('functions', {})
 
        
        buggy_functions = []
        fixed_functions = []
        

        if isinstance(functions, list):
            for func in functions:
                if isinstance(func, dict):
                    patch_function = func.get('patch_function', '')
                    if patch_function:
                        old_lines, new_lines = extract_function_from_patch(patch_function)
                        buggy_functions.append(old_lines)
                        fixed_functions.append(new_lines)
                    else:
                        print(f"Warning: Function missing patch_function field, skipping this function")
                        
                elif isinstance(func, str):
                    old_lines, new_lines = extract_function_from_patch(func)
                    buggy_functions.append(old_lines)
                    fixed_functions.append(new_lines)
        
        elif isinstance(functions, dict):
            patch_function = functions.get('patch_function', '')
            if patch_function:
                old_lines, new_lines = extract_function_from_patch(patch_function)
                buggy_functions.append(old_lines)
                fixed_functions.append(new_lines)
            else:
                 print(f"Warning: Function missing patch_function field, skipping this function")
        
        elif isinstance(functions, str):
            old_lines, new_lines = extract_function_from_patch(functions)
            buggy_functions.append(old_lines)
            fixed_functions.append(new_lines)
        
        if len(buggy_functions) != function_num or len(fixed_functions) != function_num:
            print(f"Error: buggy_functions or fixed_functions length is not equal to function_num")
            
        return buggy_functions, fixed_functions, function_num
        
        
    except Exception as e:
        print(f"Error extracting buggy/fixed code: {e}")
        import traceback
        traceback.print_exc()  
        return [], [], 0


def create_enhanced_cpg_with_versions(buggy_code, fixed_code, patch_path=None):

    buggy_ccg = None
    fixed_ccg = None

    if buggy_code and any(line.strip() for line in buggy_code):
        buggy_ccg = create_graph(buggy_code)
        if buggy_ccg and buggy_ccg.nodes():
            for node_id in buggy_ccg.nodes:
                buggy_ccg.nodes[node_id]['version'] = 'buggy'
    else:
        print("buggy_code is empty")
        print(f"patch_path: {patch_path}")
        buggy_ccg = nx.MultiDiGraph()
    
    if fixed_code and any(line.strip() for line in fixed_code):
        fixed_ccg = create_graph(fixed_code)
        
        if fixed_ccg and fixed_ccg.nodes():
            for node_id in fixed_ccg.nodes:
                fixed_ccg.nodes[node_id]['version'] = 'fixed'
    else:
        print("fixed_code is empty")
        print(f"patch_path: {patch_path}")
        fixed_ccg = nx.MultiDiGraph()

    return {
        'buggy_ccg': buggy_ccg,
        'fixed_ccg': fixed_ccg
    }


def merge_cpgs_to_mcpg(buggy_ccg, fixed_ccg):
    
    try:
        if buggy_ccg is None:
            buggy_ccg = nx.MultiDiGraph()
        if fixed_ccg is None:
            fixed_ccg = nx.MultiDiGraph()
        if not buggy_ccg.nodes() and not fixed_ccg.nodes():
            print("Warning: Both CPGs are empty, returning empty MCPG")
            return nx.MultiDiGraph()
        mcpg = nx.MultiDiGraph() 
        node_id_counter = 0
        fixed_node_mapping = {}
        buggy_node_mapping = {}
        
        for orig_node_id in fixed_ccg.nodes:
            node_data = fixed_ccg.nodes[orig_node_id].copy()
            node_data['original_id'] = orig_node_id
            node_data['version'] = 'fixed'
            mcpg.add_node(node_id_counter, **node_data)
            fixed_node_mapping[orig_node_id] = node_id_counter
            node_id_counter += 1
        
        for orig_node_id in buggy_ccg.nodes:
            node_data = buggy_ccg.nodes[orig_node_id].copy()
            node_data['original_id'] = orig_node_id
            node_data['version'] = 'buggy'
            mcpg.add_node(node_id_counter, **node_data)
            buggy_node_mapping[orig_node_id] = node_id_counter
            node_id_counter += 1        

        for u, v, key, edge_data in fixed_ccg.edges(keys=True, data=True):
            if u in fixed_node_mapping and v in fixed_node_mapping:
                new_u = fixed_node_mapping[u]
                new_v = fixed_node_mapping[v]
                edge_data_copy = edge_data.copy()
                edge_data_copy['version'] = 'fixed'
                mcpg.add_edge(new_u, new_v, key=key, **edge_data_copy)   

        for u, v, key, edge_data in buggy_ccg.edges(keys=True, data=True):
            if u in buggy_node_mapping and v in buggy_node_mapping:
                new_u = buggy_node_mapping[u]
                new_v = buggy_node_mapping[v]
                edge_data_copy = edge_data.copy()
                edge_data_copy['version'] = 'buggy'
                mcpg.add_edge(new_u, new_v, key=key, **edge_data_copy)
        common_node_map =identify_common_nodes(mcpg, fixed_node_mapping, buggy_node_mapping, fixed_ccg, buggy_ccg)
       

        
        for buggy_id, fixed_id in common_node_map.items():
           
            mcpg = nx.contracted_nodes(mcpg, fixed_id, buggy_id, self_loops=False)
            if mcpg.has_node(fixed_id):
                mcpg.nodes[fixed_id]['version'] = 'both'
        return mcpg
        
    except Exception as e:
        print(f"Error merging CPGs to MCPG: {e}")
        return nx.MultiDiGraph()


def identify_common_nodes(mcpg, fixed_node_mapping, buggy_node_mapping, fixed_ccg, buggy_ccg):
    
    try:
        common_node_map = {} # key: buggy_mcpg_id, value: fixed_mcpg_id
        def create_node_signature(ccg, node_id):
            if node_id not in ccg.nodes:
                return None
            node_data = ccg.nodes[node_id]
            return (
                node_data.get('nodeType', ''),
                node_data.get('startRow', -1),
                node_data.get('endRow', -1),
                tuple(sorted(node_data.get('sourceLines', []))),
                tuple(sorted(node_data.get('defSet', set()))),
                tuple(sorted(node_data.get('useSet', set())))
            )
        
        fixed_signatures = {node_id: create_node_signature(fixed_ccg, node_id) 
                           for node_id in fixed_ccg.nodes}
        buggy_signatures = {node_id: create_node_signature(buggy_ccg, node_id) 
                           for node_id in buggy_ccg.nodes}
        
        matched_fixed_nodes = set()
        for buggy_orig_id, buggy_sig in buggy_signatures.items():
            if buggy_sig:
                for fixed_orig_id, fixed_sig in fixed_signatures.items():
                    if fixed_orig_id not in matched_fixed_nodes and fixed_sig == buggy_sig:
                        buggy_mcpg_id = buggy_node_mapping[buggy_orig_id]
                        fixed_mcpg_id = fixed_node_mapping[fixed_orig_id]
                        
                        common_node_map[buggy_mcpg_id] = fixed_mcpg_id
                        matched_fixed_nodes.add(fixed_orig_id)
                        break
        return common_node_map

    except Exception as e:
        print(f"Error identifying common nodes: {e}")


def slice_and_prune_mcpg(mcpg):
   
    try:
        change_nodes = []
        for node_id in mcpg.nodes:
            version = mcpg.nodes[node_id].get('version', '')
            if version in ['buggy', 'fixed']:
                change_nodes.append(node_id)
        
        if not change_nodes:
            return mcpg.copy()
        sliced_nodes = set(change_nodes)
        for change_node in change_nodes:
            for successor in mcpg.successors(change_node):
                sliced_nodes.add(successor)
        
        for change_node in change_nodes:
            for predecessor in mcpg.predecessors(change_node):
                sliced_nodes.add(predecessor)
        sliced_mcpg = mcpg.subgraph(sliced_nodes).copy()        
        pruned_mcpg = prune_mcpg(sliced_mcpg)
        
        return pruned_mcpg
        
    except Exception as e:
        print(f"Error slicing and pruning MCPG: {e}")
        return mcpg.copy()


def prune_mcpg(mcpg):
   
    try:
        pruned_mcpg = mcpg.copy()
        
        isolated_nodes = list(nx.isolates(pruned_mcpg))
        pruned_mcpg.remove_nodes_from(isolated_nodes)

        empty_nodes = []
        for node_id in pruned_mcpg.nodes:
            node_data = pruned_mcpg.nodes[node_id]
            source_lines = node_data.get('sourceLines', [])
            
            if not source_lines or all(not line.strip() for line in source_lines):
                empty_nodes.append(node_id)
        
        pruned_mcpg.remove_nodes_from(empty_nodes)
        
        return pruned_mcpg
        
    except Exception as e:
        print(f"Error pruning MCPG: {e}")
        return mcpg.copy()



