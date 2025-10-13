from turtle import right
import networkx as nx
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language
from utils.utils import extract_function_from_patch
import os

# 获取 Java 语言解析器
JAVA_LANGUAGE = get_language('java')

def _extract_identifiers(node, src_lines, line_offset=0):
    
    identifiers = set()
    
    # 定义一个内部递归函数
    def find_ids(n):
        # 如果当前节点是 'identifier'，提取其文本
        if n.type == 'identifier':
            start_row, start_col = n.start_point
            end_row, end_col = n.end_point
            # 矫正行号偏移
            original_start_row = start_row - line_offset
            original_end_row = end_row - line_offset
            # 确保在单行内，以防解析错误
            if original_start_row == original_end_row and original_start_row >= 0 and original_start_row < len(src_lines):
                # .strip() to remove potential leading/trailing whitespace
                identifier_name = src_lines[original_start_row][start_col:end_col].strip()
                if identifier_name: # 确保不是空字符串
                    identifiers.add(identifier_name)
        
        # 递归遍历所有子节点
        for child in n.children:
            find_ids(child)

    find_ids(node)
    return identifiers

def java_control_dependence_graph(root_node, CCG, src_lines, parent, line_offset=0):   
        
    node_id = len(CCG.nodes)
    new_parent = parent
    
    # 定义需要加入图的节点类型
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

    # 添加图节点
    if is_graph_node:
        start_row = root_node.start_point[0] - line_offset   # 减去偏移量回到原始行号
        end_row = root_node.end_point[0] - line_offset      
        
        # 确保行号不为负数
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

    # 递归处理子节点，传递line_offset
    for child in root_node.children:
        java_control_dependence_graph(child, CCG, src_lines, new_parent, line_offset)

    if new_parent is not None:
        # 1. 处理变量定义 (defSet)
        
        # a) 局部变量声明: local_variable_declaration -> variable_declarator -> name: identifier
        if root_node.type == 'variable_declarator':
            name_node = root_node.child_by_field_name('name')
            if name_node and name_node.type == 'identifier':
                start_row, start_col = name_node.start_point
                end_row, end_col = name_node.end_point
                if start_row == end_row and start_row < len(src_lines):
                    var_name = src_lines[start_row][start_col:end_col].strip()
                    CCG.nodes[new_parent]['defSet'].add(var_name)
                    #print(f"Node {new_parent}: Added '{var_name}' to defSet (declaration)")

        # b) 赋值表达式: assignment_expression -> left: identifier
        elif root_node.type == 'assignment_expression':
            left_node = root_node.child_by_field_name('left')
            # 简单情况: a = ...
            if left_node and left_node.type == 'identifier':
                start_row, start_col = left_node.start_point
                end_row, end_col = left_node.end_point
                if start_row == end_row and start_row < len(src_lines):
                    var_name = src_lines[start_row][start_col:end_col].strip()
                    CCG.nodes[new_parent]['defSet'].add(var_name)
                    #print(f"Node {new_parent}: Added '{var_name}' to defSet (assignment)")
            # 复杂情况: this.a = ..., this.a.b = ...
            elif left_node and left_node.type == 'field_access':
                 # 找到字段访问表达式中的最右侧字段标识符
                 rightmost_field = None
                 
                 # 遍历字段访问节点的直接子节点，寻找field_identifier类型的节点
                 for child in left_node.children:
                     if child.type == 'field_identifier':
                      # field_identifier就是最右侧的字段，提取其文本
                      start_row, start_col = child.start_point
                      end_row, end_col = child.end_point
                      if start_row == end_row and start_row < len(src_lines):
                          rightmost_field  = src_lines[start_row][start_col:end_col].strip()
                          break
                    
                 if rightmost_field:
                     CCG.nodes[new_parent]['defSet'].add(rightmost_field)
                     #print(f"Node {new_parent}: Added '{rightmost_field}' to defSet (field assignment)")
                 
                 
        # c) 方法参数: formal_parameter -> name: identifier
        elif root_node.type == 'formal_parameter':
            name_node = root_node.child_by_field_name('name')
            if name_node and name_node.type == 'identifier':
                start_row, start_col = name_node.start_point
                end_row, end_col = name_node.end_point
                if start_row == end_row and start_row < len(src_lines):
                    var_name = src_lines[start_row][start_col:end_col].strip()
                    CCG.nodes[new_parent]['defSet'].add(var_name)
                    #print(f"Node {new_parent}: Added '{var_name}' to defSet (parameter)")
        
        # d) 增强 for 循环的变量: enhanced_for_statement -> 变量 identifier
        # Tree-sitter AST for 'for (Type var : collection)' has 'var' as a direct child identifier
        if root_node.type == 'enhanced_for_statement':
            # 查找声明的循环变量
            for child in root_node.children:
                if child.type == 'identifier':
                     # The first identifier after '(' is the variable
                     start_row, start_col = child.start_point
                     end_row, end_col = child.end_point
                     if start_row == end_row and start_row < len(src_lines):
                         var_name = src_lines[start_row][start_col:end_col].strip()
                         # We need to distinguish it from the collection identifier
                         # A simple heuristic: the one after the type is the variable
                         type_node = root_node.child_by_field_name('type')
                         if type_node and child.start_byte > type_node.end_byte:
                            CCG.nodes[new_parent]['defSet'].add(var_name)
                            #print(f"Node {new_parent}: Added '{var_name}' to defSet (for-each var)")
                            break # Assume only one is declared here



        use_node = None
        # a) 赋值/声明的右侧
        if root_node.type == 'variable_declarator':
            use_node = root_node.child_by_field_name('value')
        elif root_node.type == 'assignment_expression':
            use_node = root_node.child_by_field_name('right')
        # b) 控制流条件
        elif root_node.type in ['if_statement', 'while_statement']:
            use_node = root_node.child_by_field_name('condition')
        # c) for 循环的 condition 和 update 部分, 以及 collection
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

        # d) return 语句
        elif root_node.type == 'return_statement':
            use_node = root_node
        # e) 方法调用
        elif root_node.type == 'method_invocation':
            use_node = root_node
        
        # 通用提取
        if use_node:
            used_vars = _extract_identifiers(use_node, src_lines)
            # 对于赋值，要从use中移除被定义的变量
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
    
    # 查找起始节点（没有前驱的节点）
    for v in CCG.nodes:
        if len(list(CCG.predecessors(v))) == 0:
            start_nodes.append(v)
    
    start_nodes.sort()
    
    if not start_nodes:
        print("!!! DEBUG: java_control_flow_graph found NO start nodes.")
        return nx.MultiDiGraph(), []
    
    # 建立起始节点间的兄弟关系
    for i in range(len(start_nodes) - 1):
        v = start_nodes[i]
        u = start_nodes[i + 1]
        next_sibling[v] = u
    next_sibling[start_nodes[-1]] = None
    
    # 为每个节点建立其子节点间的兄弟关系和第一个子节点关系
    for v in CCG.nodes:
        children = list(CCG.neighbors(v))
        if children:
            children.sort()
            
            # 建立兄弟关系
            for i in range(len(children) - 1):
                u = children[i]
                w = children[i + 1]
                # 特殊处理：if语句的条件分支后不应该直接连接到else子句
                if (CCG.nodes[v]['nodeType'] == 'if_statement' and 
                    'clause' in CCG.nodes[w]['nodeType']):
                    next_sibling[u] = None
                else:
                    next_sibling[u] = w
            
            next_sibling[children[-1]] = None
            first_children[v] = children[0]
        else:
            first_children[v] = None
    
    # 构建控制流边
    edge_list = []
    added_edges = set()  # 用于避免重复边
    
    def add_edge_if_not_exists(from_node, to_node, edge_type='CFG'):
        """添加边，避免重复"""
        edge_key = (from_node, to_node, edge_type)
        if edge_key not in added_edges:
            edge_list.append((from_node, to_node, edge_type))
            added_edges.add(edge_key)
    
    for v in CCG.nodes:
        node_type = CCG.nodes[v]['nodeType']
        
        # 处理不同类型的节点
        if node_type in ['return_statement', 'break_statement', 'continue_statement']:
            # 终止语句：不添加后续边
            continue
            
        elif node_type in ['for_statement', 'while_statement']:
            # 循环语句：条件->循环体->条件（回边）
            first_child = first_children.get(v)
            if first_child is not None:
                add_edge_if_not_exists(v, first_child)
            
            # 循环的下一个兄弟节点（循环结束后执行）
            next_sib = next_sibling.get(v)
            if next_sib is not None:
                add_edge_if_not_exists(v, next_sib)
                
        elif node_type == 'if_statement':
            # if语句：条件->then分支，条件->else分支（如果存在）
            for child in CCG.neighbors(v):
                child_type = CCG.nodes[child]['nodeType']
                if child_type in ['if_statement', 'else_clause'] or 'clause' in child_type:
                    add_edge_if_not_exists(v, child)
            
            # if语句的第一个子节点（通常是条件表达式）
            first_child = first_children.get(v)
            if first_child is not None:
                add_edge_if_not_exists(v, first_child)
                
        elif node_type == 'try_statement':
            # try语句：try块->catch块->finally块
            for child in CCG.neighbors(v):
                child_type = CCG.nodes[child]['nodeType']
                if child_type in ['try_statement', 'catch_clause', 'finally_clause']:
                    add_edge_if_not_exists(v, child)
            
            first_child = first_children.get(v)
            if first_child is not None:
                add_edge_if_not_exists(v, first_child)
                
        elif 'clause' in node_type:
            # 处理各种clause节点
            first_child = first_children.get(v)
            if first_child is not None:
                add_edge_if_not_exists(v, first_child)
                
        else:
            # 普通语句：顺序执行
            first_child = first_children.get(v)
            if first_child is not None:
                add_edge_if_not_exists(v, first_child)
        
        # 添加到下一个兄弟节点的边（顺序执行）
        next_sib = next_sibling.get(v)
        if next_sib is not None:
            add_edge_if_not_exists(v, next_sib)
    
    # 构建CFG
    CFG.add_edges_from(edge_list)
    
    # 确保所有节点都在CFG中，并复制节点属性
    for v in CCG.nodes:
        if v not in CFG.nodes:
            CFG.add_node(v)
        
        # 复制CCG节点的属性到CFG
        if v in CFG.nodes:
            CFG.nodes[v].update(CCG.nodes[v])
    
    # 添加调试信息
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
    
    # 保证 code_lines 是每行一条的列表
    if isinstance(code_lines, str):
        code_lines = code_lines.splitlines()
    elif isinstance(code_lines, list) and len(code_lines) == 1 and '\n' in code_lines[0]:
        code_lines = code_lines[0].splitlines()
    
    # 检查代码片段类型并进行动态包裹
    line_offset = 0
    is_function_level = True
    
    # 检查首行是否以特定关键字开头，如果是则视为文件级代码
    if code_lines:
        first_line = code_lines[0].strip()
        # 去除diff标志后再判断
        if first_line.startswith('+') or first_line.startswith('-'):
            first_line = first_line[1:].strip()
        if (first_line.startswith('package') or 
            first_line.startswith('import')):
            is_function_level = False
            # print(f"检测到文件级代码，首行: {first_line}")
    
    # 如果是函数级代码，添加虚拟类包裹
    if is_function_level:
        #print("检测到函数级代码，添加虚拟类包裹")
        wrapped_lines = ['class DummyWrapper {'] + code_lines + ['}']
        code_lines = wrapped_lines
        line_offset = 1  # 记录偏移量
    #else:
        #print("文件级代码，无需包裹")
    
    src_lines = [line + '\n' for line in code_lines ]  
    #print("src_lines:", src_lines)
    

    # 初始化解析器
    parser = Parser()
    JAVA_LANGUAGE = None

    # 解析方式 1：使用 tree_sitter_languages 加载 Java 语言
    try:
        JAVA_LANGUAGE = get_language('java')
        if JAVA_LANGUAGE is not None:
            parser.set_language(JAVA_LANGUAGE)
            # print("Using tree_sitter_languages to load Java parser")
        else:
            raise Exception("Failed to load Java language from tree_sitter_languages")
    except Exception as e:
        print(f"tree_sitter_languages failed: {e}")
        # 解析方式 2：回退到使用 tree_sitter 的传统方式
        try:
            # 在 Windows 上，动态链接库文件后缀为 .dll
            language_path = './my-languages.dll' if os.name == 'nt' else './my-languages.so'
            Language.build_library(language_path, ['./tree-sitter-java'])
            JAVA_LANGUAGE = Language(language_path, 'java')
            parser.set_language(JAVA_LANGUAGE)
            # print("Using fallback tree_sitter language loading method")
        except Exception as e:
            print(f"Fallback method failed: {e}")
            return None
    
    # 如果代码行为空，返回 None
    if len(src_lines) == 0:
        print("src_lines is empty")
        return None
    
    # 处理注释（单行 // 和多行 /* */）
    comment_lines = []
    in_multiline_comment = False
    for i, line in enumerate(src_lines):
        #original_line = line
        line = line.lstrip()

        # 如果当前处于多行注释中
        if in_multiline_comment:
            if '*/' in line:
                # 找到结束符，只去掉注释部分
                before, after = line.split('*/', 1)
                src_lines[i] = after if after.strip() else '\n'
                in_multiline_comment = False
            else:
                # 整行都在注释中
                comment_lines.append(i)
                src_lines[i] = '\n'
        else:
            # 行内或整行 // 注释
            if '//' in line:
                code, _ = line.split('//', 1)
                if code.strip():  # 注释前有代码
                    src_lines[i] = code.rstrip() + '\n'
                else:  # 整行都是注释
                    src_lines[i] = '\n'
                comment_lines.append(i)

            # 行内或多行 /*...*/ 注释
            elif '/*' in line:
                before, after = line.split('/*', 1)
                if '*/' in after:
                    # 单行注释：/* ... */
                    _, after_comment = after.split('*/', 1)
                    src_lines[i] = before + after_comment
                else:
                    # 多行注释开始
                    src_lines[i] = before if before.strip() else '\n'
                    in_multiline_comment = True
                comment_lines.append(i)
    #print("comment_lines:", comment_lines)
    
    
    # 定义读取代码的回调函数，用于解析器
    def read_callable(byte_offset, point):
        row, column = point
        if row >= len(src_lines) or column >= len(src_lines[row]):
            return None
        return src_lines[row][column:].encode('utf8', errors='ignore')

    # 使用解析器解析代码，生成语法树
    tree = parser.parse(read_callable)
    
    
    # 如果是函数级代码，从class_body开始构建图
    root_nodes = []
    if is_function_level and tree.root_node.children:
        # 寻找class_declaration -> class_body
        for child in tree.root_node.children:
            if child.type == 'class_declaration':
                for subchild in child.children:
                    if subchild.type == 'class_body':
                        # 从class_body的子节点开始
                        root_nodes = list(subchild.children)
                        break
                break
    else:
        root_nodes = list(tree.root_node.children)
    
    
    # 检查是否所有子节点都是注释
    all_comment = True
    for child in root_nodes:
        if child.type != 'comment':
            all_comment = False
    if all_comment:
        return None

    # 初始化程序依赖图（多重有向图）
    ccg = nx.MultiDiGraph()

   
    # # 构建 Java 控制依赖边，传递line_offset进行行号矫正
    for child in root_nodes:
         java_control_dependence_graph(child, ccg, code_lines, None, line_offset)

    # 构建 Java 控制流图
    cfg, cfg_edge_list = java_control_flow_graph(ccg)
   
    # 构建 Java 数据依赖图
    java_data_dependence_graph(cfg, ccg)

    # 将控制流边添加到程序依赖图
    
    ccg.add_edges_from(cfg_edge_list)

    # 处理注释行，调整节点行号和源代码行
    node_list = list(ccg.nodes)
    node_list.sort()
    comment_lines.reverse()
    max_comment_line = 0
    
    # 使用原始代码行进行注释处理
   

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
    # 返回最终的程序依赖图
    return ccg



def extract_patch_subgraph(old_ccg, new_ccg):
    """从旧版本和新版本的PDG中提取补丁变更子图"""
    if old_ccg is None:
        old_ccg = nx.MultiDiGraph()
    if new_ccg is None:
        new_ccg = nx.MultiDiGraph()
    
    # 创建节点映射（基于代码行号和节点类型）
    def create_node_signature(ccg, node_id):
        if node_id not in ccg.nodes:
            return None
        node_data = ccg.nodes[node_id]
         # 返回节点的签名，包含节点类型、起始行、结束行、定义集和使用集
        return (
            node_data.get('nodeType', ''),
            node_data.get('startRow', -1),
            node_data.get('endRow', -1),
            tuple(sorted(node_data.get('defSet', set()))),
            tuple(sorted(node_data.get('useSet', set())))
        )
    
    # 构建节点签名映射
    old_signatures = {node_id: create_node_signature(old_ccg, node_id) 
                     for node_id in old_ccg.nodes}
    new_signatures = {node_id: create_node_signature(new_ccg, node_id) 
                     for node_id in new_ccg.nodes}
    
    # 反向映射：签名到节点ID
    old_sig_to_node = {sig: node_id for node_id, sig in old_signatures.items() if sig}
    new_sig_to_node = {sig: node_id for node_id, sig in new_signatures.items() if sig}
    
    # 获取旧版本和新版本的签名集合
    old_sigs = set(old_sig_to_node.keys())
    new_sigs = set(new_sig_to_node.keys())
    
    # 找出新增、删除的节点
    added_sigs = new_sigs - old_sigs
    removed_sigs = old_sigs - new_sigs
    
    added_nodes = [new_sig_to_node[sig] for sig in added_sigs]
    removed_nodes = [old_sig_to_node[sig] for sig in removed_sigs]
    
    # 创建变更子图
    subgraph = nx.MultiDiGraph()
    
    # 添加所有相关节点到子图
    relevant_nodes = set()
    
    # 添加直接变更的节点
    for node_id in added_nodes:
        if node_id in new_ccg.nodes:
            relevant_nodes.add(('new', node_id))
    
    for node_id in removed_nodes:
        if node_id in old_ccg.nodes:
            relevant_nodes.add(('old', node_id))
    
    # 添加变更节点的邻居节点（上下文）
    for version, node_id in list(relevant_nodes):
        ccg = new_ccg if version == 'new' else old_ccg
        # 添加前驱和后继节点
        for neighbor in list(ccg.predecessors(node_id)) + list(ccg.successors(node_id)):
            relevant_nodes.add((version, neighbor))
    
    # 构建子图
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
    
    # 添加相关边
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
    
    # 收集所有变量名
    all_variables = set()
    for node_id in normalized_graph.nodes:
        node_data = normalized_graph.nodes[node_id]
        all_variables.update(node_data.get('defSet', set()))
        all_variables.update(node_data.get('useSet', set()))
    
    # 创建变量名映射
    var_mapping = {}
    var_counter = 1
    for var in sorted(all_variables):
        if var and var.isidentifier():
            var_mapping[var] = f'var{var_counter}'
            var_counter += 1
    
    # 应用变量名映射
    for node_id in normalized_graph.nodes:
        node_data = normalized_graph.nodes[node_id]
        
        # 规范化defSet
        if 'defSet' in node_data:
            new_def_set = set()
            for var in node_data['defSet']:
                new_def_set.add(var_mapping.get(var, var))
            normalized_graph.nodes[node_id]['defSet'] = new_def_set
        
        # 规范化useSet
        if 'useSet' in node_data:
            new_use_set = set()
            for var in node_data['useSet']:
                new_use_set.add(var_mapping.get(var, var))
            normalized_graph.nodes[node_id]['useSet'] = new_use_set
        
        # 规范化源代码行（移除多余空白）
        if 'sourceLines' in node_data:
            normalized_lines = []
            for line in node_data['sourceLines']:
                # 规范化空白
                line = ' '.join(line.split())
                if line:
                    normalized_lines.append(line)
            normalized_graph.nodes[node_id]['sourceLines'] = normalized_lines
    
    return normalized_graph


def create_mcpg_from_patch(patch_data): 

    try:
        content = patch_data.get('content', {})
        function_num = content.get('function_num', 1)

        # 提取patch_path用于错误定位
        patch_path = patch_data.get('patch_path', 'Unknown path')
        
        buggy_code_list, fixed_code_list, _ = extract_enhanced_buggy_fixed_code(patch_data)


        function_mcpgs = []

        # 确保两个列表长度一致，防止索引越界
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
                # 如果两个CPG都为空，跳过这个函数处理
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
        # 出现异常时，也返回一个结构，以便调用者可以检查错误
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
            # 处理函数列表
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
                    # 如果函数是字符串格式的diff
                    old_lines, new_lines = extract_function_from_patch(func)
                    buggy_functions.append(old_lines)
                    fixed_functions.append(new_lines)
        
        elif isinstance(functions, dict):
            # 如果functions是单个字典（用户提供的格式）
            patch_function = functions.get('patch_function', '')
            if patch_function:
                old_lines, new_lines = extract_function_from_patch(patch_function)
                buggy_functions.append(old_lines)
                fixed_functions.append(new_lines)
            else:
                 print(f"Warning: Function missing patch_function field, skipping this function")
        
        elif isinstance(functions, str):
            # 如果functions是单个字符串
            old_lines, new_lines = extract_function_from_patch(functions)
            buggy_functions.append(old_lines)
            fixed_functions.append(new_lines)
        
        # 确保返回的函数数量与function_num一致
        if len(buggy_functions) != function_num or len(fixed_functions) != function_num:
            print(f"Error: buggy_functions or fixed_functions length is not equal to function_num")
            
        return buggy_functions, fixed_functions, function_num
        
        
    except Exception as e:
        print(f"Error extracting buggy/fixed code: {e}")
        import traceback
        traceback.print_exc()  # 添加详细的错误追踪
        return [], [], 0


def create_enhanced_cpg_with_versions(buggy_code, fixed_code, patch_path=None):

    buggy_ccg = None
    fixed_ccg = None

    # 直接使用原始代码创建图
    if buggy_code and any(line.strip() for line in buggy_code):
        buggy_ccg = create_graph(buggy_code)
        # 为所有节点添加版本信息
        if buggy_ccg and buggy_ccg.nodes():
            for node_id in buggy_ccg.nodes:
                buggy_ccg.nodes[node_id]['version'] = 'buggy'
    else:
        # 如果代码无效，直接创建一个空的图，避免调用 create_graph
        print("buggy_code is empty")
        print(f"patch_path: {patch_path}")
        buggy_ccg = nx.MultiDiGraph()
    
    if fixed_code and any(line.strip() for line in fixed_code):
        fixed_ccg = create_graph(fixed_code)
        
        # 为所有节点添加版本信息
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
        # 输入验证和规范化
        if buggy_ccg is None:
            buggy_ccg = nx.MultiDiGraph()
        if fixed_ccg is None:
            fixed_ccg = nx.MultiDiGraph()
            
        # 如果两个CPG都为空，返回空MCPG
        if not buggy_ccg.nodes() and not fixed_ccg.nodes():
            print("Warning: Both CPGs are empty, returning empty MCPG")
            return nx.MultiDiGraph()
        mcpg = nx.MultiDiGraph()
        
        # 重新分配节点ID以避免冲突
        # 先添加fixed节点，再添加buggy节点
        node_id_counter = 0
        fixed_node_mapping = {}
        buggy_node_mapping = {}
        
        # 添加fixed节点
        for orig_node_id in fixed_ccg.nodes:
            # 复制节点的属性数据
            node_data = fixed_ccg.nodes[orig_node_id].copy()
            node_data['original_id'] = orig_node_id
            # 添加版本信息
            node_data['version'] = 'fixed'
            # 添加节点
            mcpg.add_node(node_id_counter, **node_data)
            fixed_node_mapping[orig_node_id] = node_id_counter
            node_id_counter += 1
        
        # 添加buggy节点
        for orig_node_id in buggy_ccg.nodes:
            node_data = buggy_ccg.nodes[orig_node_id].copy()
            node_data['original_id'] = orig_node_id
            node_data['version'] = 'buggy'
            mcpg.add_node(node_id_counter, **node_data)
            buggy_node_mapping[orig_node_id] = node_id_counter
            node_id_counter += 1
        
        # 添加fixed边
        for u, v, key, edge_data in fixed_ccg.edges(keys=True, data=True):
            if u in fixed_node_mapping and v in fixed_node_mapping:
                new_u = fixed_node_mapping[u]
                new_v = fixed_node_mapping[v]
                edge_data_copy = edge_data.copy()
                edge_data_copy['version'] = 'fixed'
                mcpg.add_edge(new_u, new_v, key=key, **edge_data_copy)
        
        # 添加buggy边
        for u, v, key, edge_data in buggy_ccg.edges(keys=True, data=True):
            if u in buggy_node_mapping and v in buggy_node_mapping:
                new_u = buggy_node_mapping[u]
                new_v = buggy_node_mapping[v]
                edge_data_copy = edge_data.copy()
                edge_data_copy['version'] = 'buggy'
                mcpg.add_edge(new_u, new_v, key=key, **edge_data_copy)
        
        # 识别共同节点（未发生变化的节点）
        common_node_map =identify_common_nodes(mcpg, fixed_node_mapping, buggy_node_mapping, fixed_ccg, buggy_ccg)
       
        #合并共同节点
        
        for buggy_id, fixed_id in common_node_map.items():
           
            mcpg = nx.contracted_nodes(mcpg, fixed_id, buggy_id, self_loops=False)
            
            # 在合并后，我们要确保保留的节点的版本是 'both'
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
        
        # 构建签名映射
        fixed_signatures = {node_id: create_node_signature(fixed_ccg, node_id) 
                           for node_id in fixed_ccg.nodes}
        buggy_signatures = {node_id: create_node_signature(buggy_ccg, node_id) 
                           for node_id in buggy_ccg.nodes}
        
        matched_fixed_nodes = set()
        # 找出相同的节点
        for buggy_orig_id, buggy_sig in buggy_signatures.items():
            if buggy_sig:
                for fixed_orig_id, fixed_sig in fixed_signatures.items():
                    if fixed_orig_id not in matched_fixed_nodes and fixed_sig == buggy_sig:
                        # 标记为共同节点
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
        # 找出变更节点作为切片起点
        change_nodes = []
        for node_id in mcpg.nodes:
            version = mcpg.nodes[node_id].get('version', '')
            if version in ['buggy', 'fixed']:
                change_nodes.append(node_id)
        
        if not change_nodes:
            return mcpg.copy()
        
        # 执行切片：找出与变更节点相关的所有节点
        sliced_nodes = set(change_nodes)
        
        # 正向切片：找出依赖变更节点的节点
        for change_node in change_nodes:
            # 添加后继节点（单跳）
            for successor in mcpg.successors(change_node):
                sliced_nodes.add(successor)
        
        # 反向切片：找出影响变更节点的节点  
        for change_node in change_nodes:
            # 添加前驱节点（单跳）
            for predecessor in mcpg.predecessors(change_node):
                sliced_nodes.add(predecessor)
        
        # 创建切片子图
        sliced_mcpg = mcpg.subgraph(sliced_nodes).copy()
        
        # 剪枝：删除孤立节点和空代码节点
        pruned_mcpg = prune_mcpg(sliced_mcpg)
        
        return pruned_mcpg
        
    except Exception as e:
        print(f"Error slicing and pruning MCPG: {e}")
        return mcpg.copy()


def prune_mcpg(mcpg):
   
    try:
        pruned_mcpg = mcpg.copy()
        
        # 删除孤立节点
        isolated_nodes = list(nx.isolates(pruned_mcpg))
        pruned_mcpg.remove_nodes_from(isolated_nodes)
        
        # 删除空代码节点
        empty_nodes = []
        for node_id in pruned_mcpg.nodes:
            node_data = pruned_mcpg.nodes[node_id]
            source_lines = node_data.get('sourceLines', [])
            
            # 检查是否为空代码节点
            if not source_lines or all(not line.strip() for line in source_lines):
                empty_nodes.append(node_id)
        
        pruned_mcpg.remove_nodes_from(empty_nodes)
        
        return pruned_mcpg
        
    except Exception as e:
        print(f"Error pruning MCPG: {e}")
        return mcpg.copy()



# 备用函数
def create_patch_graph(old_code, new_code):
    """从旧版本和新版本代码创建补丁变更图"""
    try:
        # 生成旧版本和新版本的PDG
        old_ccg = create_graph(old_code) if old_code else None
        new_ccg = create_graph(new_code) if new_code else None
        
        # 提取补丁变更
        patch_info = extract_patch_subgraph(old_ccg, new_ccg)
        
        # 规范化子图    
        normalized_subgraph = normalize_subgraph(patch_info['subgraph'])
        
        return {
            'old_ccg': old_ccg,
            'new_ccg': new_ccg,
            'patch_info': patch_info,
            'normalized_subgraph': normalized_subgraph
        }
        
    except Exception as e:
        print(f"Error creating patch graph: {e}")
        return {
            'old_ccg': None,
            'new_ccg': None,
            'patch_info': None,
            'normalized_subgraph': nx.MultiDiGraph()
        }


