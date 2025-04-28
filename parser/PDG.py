from tree_sitter import Language, Parser
from .utils import (
    remove_comments_and_docstrings,
    tree_to_token_index,
    index_to_code_token,
    tree_to_variable_index
)

def PDG_python(root_node, index_to_code, states):
    assignment = ['assignment', 'augmented_assignment', 'for_in_clause']
    if_statement = ['if_statement']
    for_statement = ['for_statement']
    while_statement = ['while_statement']
    do_first_statement = ['for_in_clause']
    def_statement = ['default_parameter']
    control_flow_nodes = ['if_statement', 'for_statement', 'while_statement']

    states = states.copy()
    pdg = []

    def add_control_dependency(node, parent_node):
        """
        添加控制依赖关系
        """
        if parent_node:
            pdg.append((node, parent_node, 'controlDependsOn'))

    def add_data_dependency(node, parent_node):
        """
        添加数据依赖关系
        """
        if parent_node:
            pdg.append((node, parent_node, 'dataDependsOn'))

    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        idx, code = index_to_code[(root_node.start_point, root_node.end_point)]
        if root_node.type == code:
            return [], states
        elif code in states:
            return [(code, idx, 'comesFrom', [code], states[code].copy())], states
        else:
            if root_node.type == 'identifier':
                states[code] = [idx]
            return [(code, idx, 'comesFrom', [], [])], states

    elif root_node.type in def_statement:
        name = root_node.child_by_field_name('name')
        value = root_node.child_by_field_name('value')
        if value is None:
            indexs = tree_to_variable_index(name, index_to_code)
            for index in indexs:
                idx, code = index_to_code[index]
                pdg.append((code, idx, 'comesFrom', [], []))
                states[code] = [idx]
            return sorted(pdg, key=lambda x: x[1]), states
        else:
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            temp, states = PDG_python(value, index_to_code, states)
            pdg += temp
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in assignment:
        if root_node.type == 'for_in_clause':
            right_nodes = [root_node.children[-1]]
            left_nodes = [root_node.child_by_field_name('left')]
        else:
            left_nodes = [x for x in root_node.child_by_field_name('left').children if x.type != ',']
            right_nodes = [x for x in root_node.child_by_field_name('right').children if x.type != ',']
            if len(right_nodes) != len(left_nodes):
                left_nodes = [root_node.child_by_field_name('left')]
                right_nodes = [root_node.child_by_field_name('right')]
            if len(left_nodes) == 0:
                left_nodes = [root_node.child_by_field_name('left')]
            if len(right_nodes) == 0:
                right_nodes = [root_node.child_by_field_name('right')]
        for node in right_nodes:
            temp, states = PDG_python(node, index_to_code, states)
            pdg += temp

        for left_node, right_node in zip(left_nodes, right_nodes):
            left_tokens_index = tree_to_variable_index(left_node, index_to_code)
            right_tokens_index = tree_to_variable_index(right_node, index_to_code)
            for token1_index in left_tokens_index:
                idx1, code1 = index_to_code[token1_index]
                for token2_index in right_tokens_index:
                    idx2, code2 = index_to_code[token2_index]
                    pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                states[code1] = [idx1]

        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in if_statement:
        pdg = []
        current_states = states.copy()
        others_states = []
        tag = False
        if 'else' in root_node.type:
            tag = True
        for child in root_node.children:
            if 'else' in child.type:
                tag = True
            if child.type not in ['elif_clause', 'else_clause']:
                temp, current_states = PDG_python(child, index_to_code, current_states)
                pdg += temp
            else:
                temp, new_states = PDG_python(child, index_to_code, states)
                pdg += temp
                others_states.append(new_states)
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)
        new_states = {}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(pdg, key=lambda x: x[1]), new_states

    elif root_node.type in for_statement:
        pdg = []
        for child in root_node.children:
            temp, states = PDG_python(child, index_to_code, states)
            pdg += temp
        flag = False
        for child in root_node.children:
            if flag:
                temp, states = PDG_python(child, index_to_code, states)
                pdg += temp
            elif child.type == "local_variable_declaration":
                flag = True
        dic = {}
        for x in pdg:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(set(dic[(x[0], x[1], x[2])][0] + x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(list(set(dic[(x[0], x[1], x[2])][1] + x[4])))
        pdg = [(x[0], x[1], x[2], y[0], y[1]) for x, y in sorted(dic.items(), key=lambda t: t[0][1])]
        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in while_statement:
        pdg = []
        for i in range(2):
            for child in root_node.children:
                temp, states = PDG_python(child, index_to_code, states)
                pdg += temp
        dic = {}
        for x in pdg:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(set(dic[(x[0], x[1], x[2])][0] + x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(list(set(dic[(x[0], x[1], x[2])][1] + x[4])))
        pdg = [(x[0], x[1], x[2], y[0], y[1]) for x, y in sorted(dic.items(), key=lambda t: t[0][1])]
        return sorted(pdg, key=lambda x: x[1]), states

    else:
        pdg = []
        for child in root_node.children:
            if child.type in do_first_statement:
                temp, states = PDG_python(child, index_to_code, states)
                pdg += temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp, states = PDG_python(child, index_to_code, states)
                pdg += temp
        return sorted(pdg, key=lambda x: x[1]), states

def PDG_java(root_node, index_to_code, states):
    assignment = ['assignment_expression']
    def_statement = ['variable_declarator']
    increment_statement = ['update_expression']
    if_statement = ['if_statement', 'else']
    for_statement = ['for_statement']
    enhanced_for_statement = ['enhanced_for_statement']
    while_statement = ['while_statement']
    do_first_statement = []
    control_flow_nodes = ['if_statement', 'for_statement', 'while_statement', 'enhanced_for_statement']

    states = states.copy()
    pdg = []

    def add_control_dependency(node, parent_node):
        """
        添加控制依赖关系
        """
        if parent_node:
            pdg.append((node, parent_node, 'controlDependsOn'))

    def add_data_dependency(node, parent_node):
        """
        添加数据依赖关系
        """
        if parent_node:
            pdg.append((node, parent_node, 'dataDependsOn'))

    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        idx, code = index_to_code[(root_node.start_point, root_node.end_point)]
        if root_node.type == code:
            return [], states
        elif code in states:
            return [(code, idx, 'comesFrom', [code], states[code].copy())], states
        else:
            if root_node.type == 'identifier':
                states[code] = [idx]
            return [(code, idx, 'comesFrom', [], [])], states

    elif root_node.type in def_statement:
        name = root_node.child_by_field_name('name')
        value = root_node.child_by_field_name('value')
        if value is None:
            indexs = tree_to_variable_index(name, index_to_code)
            for index in indexs:
                idx, code = index_to_code[index]
                pdg.append((code, idx, 'comesFrom', [], []))
                states[code] = [idx]
            return sorted(pdg, key=lambda x: x[1]), states
        else:
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            temp, states = PDG_java(value, index_to_code, states)
            pdg += temp
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in assignment:
        left_nodes = root_node.child_by_field_name('left')
        right_nodes = root_node.child_by_field_name('right')
        temp, states = PDG_java(right_nodes, index_to_code, states)
        pdg += temp
        name_indexs = tree_to_variable_index(left_nodes, index_to_code)
        value_indexs = tree_to_variable_index(right_nodes, index_to_code)
        for index1 in name_indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in value_indexs:
                idx2, code2 = index_to_code[index2]
                pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            states[code1] = [idx1]
        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in increment_statement:
        pdg = []
        indexs = tree_to_variable_index(root_node, index_to_code)
        for index1 in indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in indexs:
                idx2, code2 = index_to_code[index2]
                pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            states[code1] = [idx1]
        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in if_statement:
        pdg = []
        current_states = states.copy()
        others_states = []
        tag = False
        if 'else' in root_node.type:
            tag = True
        for child in root_node.children:
            if 'else' in child.type:
                tag = True
            if child.type not in if_statement and not tag:
                temp, current_states = PDG_java(child, index_to_code, current_states)
                pdg += temp
            else:
                temp, new_states = PDG_java(child, index_to_code, states)
                pdg += temp
                others_states.append(new_states)
        others_states.append(current_states)
        if not tag:
            others_states.append(states)
        new_states = {}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(pdg, key=lambda x: x[1]), new_states

    elif root_node.type in for_statement:
        pdg = []
        for child in root_node.children:
            temp, states = PDG_java(child, index_to_code, states)
            pdg += temp
        flag = False
        for child in root_node.children:
            if flag:
                temp, states = PDG_java(child, index_to_code, states)
                pdg += temp
            elif child.type == "local_variable_declaration":
                flag = True
        # 添加控制依赖（循环体依赖循环条件）
        for child in root_node.children:
            if child.type == "condition":
                condition_code = index_to_code[(child.start_point, child.end_point)][1]
                for body_child in root_node.children:
                    if body_child.type == "statement":
                        body_code = index_to_code[(body_child.start_point, body_child.end_point)][1]
                        pdg.append((body_code, condition_code, 'controlDependsOn'))
        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in enhanced_for_statement:
        name = root_node.child_by_field_name('name')
        value = root_node.child_by_field_name('value')
        body = root_node.child_by_field_name('body')
        for i in range(2):
            temp, states = PDG_java(value, index_to_code, states)
            pdg += temp
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                states[code1] = [idx1]
            temp, states = PDG_java(body, index_to_code, states)
            pdg += temp
        # 添加控制依赖（循环体依赖循环条件）
        value_code = index_to_code[(value.start_point, value.end_point)][1]
        for body_child in body.children:
            body_code = index_to_code[(body_child.start_point, body_child.end_point)][1]
            pdg.append((body_code, value_code, 'controlDependsOn'))
        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in while_statement:
        pdg = []
        for i in range(2):
            for child in root_node.children:
                temp, states = PDG_java(child, index_to_code, states)
                pdg += temp
        # 添加控制依赖（循环体依赖循环条件）
        condition = root_node.child_by_field_name('condition')
        condition_code = index_to_code[(condition.start_point, condition.end_point)][1]
        body = root_node.child_by_field_name('body')
        for body_child in body.children:
            body_code = index_to_code[(body_child.start_point, body_child.end_point)][1]
            pdg.append((body_code, condition_code, 'controlDependsOn'))
        return sorted(pdg, key=lambda x: x[1]), states

def PDG_csharp(root_node, index_to_code, states):
    """
    生成C#代码的程序依赖图（PDG）
    """
    assignment = ['assignment_expression']
    def_statement = ['variable_declarator']
    increment_statement = ['postfix_unary_expression']
    if_statement = ['if_statement', 'else']
    for_statement = ['for_statement']
    enhanced_for_statement = ['for_each_statement']
    while_statement = ['while_statement']
    do_first_statement = []
    control_flow_nodes = ['if_statement', 'for_statement', 'while_statement', 'enhanced_for_statement']

    states = states.copy()
    pdg = []

    def add_control_dependency(node, parent_node):
        """
        添加控制依赖关系
        """
        if parent_node:
            pdg.append((node, parent_node, 'controlDependsOn'))

    def add_data_dependency(node, parent_node):
        """
        添加数据依赖关系
        """
        if parent_node:
            pdg.append((node, parent_node, 'dataDependsOn'))

    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        idx, code = index_to_code[(root_node.start_point, root_node.end_point)]
        if root_node.type == code:
            return [], states
        elif code in states:
            return [(code, idx, 'comesFrom', [code], states[code].copy())], states
        else:
            if root_node.type == 'identifier':
                states[code] = [idx]
            return [(code, idx, 'comesFrom', [], [])], states

    elif root_node.type in def_statement:
        if len(root_node.children) == 2:
            name = root_node.children[0]
            value = root_node.children[1]
        else:
            name = root_node.children[0]
            value = None
        if value is None:
            indexs = tree_to_variable_index(name, index_to_code)
            for index in indexs:
                idx, code = index_to_code[index]
                pdg.append((code, idx, 'comesFrom', [], []))
                states[code] = [idx]
            return sorted(pdg, key=lambda x: x[1]), states
        else:
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            temp, states = PDG_csharp(value, index_to_code, states)
            pdg += temp
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in assignment:
        left_nodes = root_node.child_by_field_name('left')
        right_nodes = root_node.child_by_field_name('right')
        temp, states = PDG_csharp(right_nodes, index_to_code, states)
        pdg += temp
        name_indexs = tree_to_variable_index(left_nodes, index_to_code)
        value_indexs = tree_to_variable_index(right_nodes, index_to_code)
        for index1 in name_indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in value_indexs:
                idx2, code2 = index_to_code[index2]
                pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            states[code1] = [idx1]
        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in increment_statement:
        pdg = []
        indexs = tree_to_variable_index(root_node, index_to_code)
        for index1 in indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in indexs:
                idx2, code2 = index_to_code[index2]
                pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            states[code1] = [idx1]
        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in if_statement:
        pdg = []
        current_states = states.copy()
        others_states = []
        tag = False
        if 'else' in root_node.type:
            tag = True
        for child in root_node.children:
            if 'else' in child.type:
                tag = True
            if child.type not in if_statement and not tag:
                temp, current_states = PDG_csharp(child, index_to_code, current_states)
                pdg += temp
            else:
                temp, new_states = PDG_csharp(child, index_to_code, states)
                pdg += temp
                others_states.append(new_states)
        others_states.append(current_states)
        if not tag:
            others_states.append(states)
        new_states = {}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(pdg, key=lambda x: x[1]), new_states

    elif root_node.type in for_statement:
        pdg = []
        for child in root_node.children:
            temp, states = PDG_csharp(child, index_to_code, states)
            pdg += temp
        flag = False
        for child in root_node.children:
            if flag:
                temp, states = PDG_csharp(child, index_to_code, states)
                pdg += temp
            elif child.type == "local_variable_declaration":
                flag = True
        # 添加控制依赖（循环体依赖循环条件）
        condition = root_node.child_by_field_name('condition')
        if condition:
            condition_code = index_to_code[(condition.start_point, condition.end_point)][1]
            body = root_node.child_by_field_name('body')
            for body_child in body.children:
                body_code = index_to_code[(body_child.start_point, body_child.end_point)][1]
                pdg.append((body_code, condition_code, 'controlDependsOn'))
        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in enhanced_for_statement:
        name = root_node.child_by_field_name('left')
        value = root_node.child_by_field_name('right')
        body = root_node.child_by_field_name('body')
        for i in range(2):
            temp, states = PDG_csharp(value, index_to_code, states)
            pdg += temp
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                states[code1] = [idx1]
            temp, states = PDG_csharp(body, index_to_code, states)
            pdg += temp
        # 添加控制依赖（循环体依赖循环条件）
        value_code = index_to_code[(value.start_point, value.end_point)][1]
        for body_child in body.children:
            body_code = index_to_code[(body_child.start_point, body_child.end_point)][1]
            pdg.append((body_code, value_code, 'controlDependsOn'))
        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in while_statement:
        pdg = []
        for i in range(2):
            for child in root_node.children:
                temp, states = PDG_csharp(child, index_to_code, states)
                pdg += temp
        # 添加控制依赖（循环体依赖循环条件）
        condition = root_node.child_by_field_name('condition')
        if condition:
            condition_code = index_to_code[(condition.start_point, condition.end_point)][1]
            body = root_node.child_by_field_name('body')
            for body_child in body.children:
                body_code = index_to_code[(body_child.start_point, body_child.end_point)][1]
                pdg.append((body_code, condition_code, 'controlDependsOn'))
        return sorted(pdg, key=lambda x: x[1]), states

    else:
        pdg = []
        for child in root_node.children:
            if child.type in do_first_statement:
                temp, states = PDG_csharp(child, index_to_code, states)
                pdg += temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp, states = PDG_csharp(child, index_to_code, states)
                pdg += temp
        return sorted(pdg, key=lambda x: x[1]), states

def PDG_ruby(root_node, index_to_code, states):
    """
    生成Ruby代码的程序依赖图（PDG）
    """
    assignment = ['assignment', 'operator_assignment']
    if_statement = ['if', 'elsif', 'else', 'unless', 'when']
    for_statement = ['for']
    while_statement = ['while_modifier', 'until']
    do_first_statement = []
    def_statement = ['keyword_parameter']
    control_flow_nodes = ['if', 'elsif', 'else', 'unless', 'when', 'for', 'while_modifier', 'until']

    states = states.copy()
    pdg = []

    def add_control_dependency(node, parent_node):
        """
        添加控制依赖关系
        """
        if parent_node:
            pdg.append((node, parent_node, 'controlDependsOn'))

    def add_data_dependency(node, parent_node):
        """
        添加数据依赖关系
        """
        if parent_node:
            pdg.append((node, parent_node, 'dataDependsOn'))

    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        idx, code = index_to_code[(root_node.start_point, root_node.end_point)]
        if root_node.type == code:
            return [], states
        elif code in states:
            return [(code, idx, 'comesFrom', [code], states[code].copy())], states
        else:
            if root_node.type == 'identifier':
                states[code] = [idx]
            return [(code, idx, 'comesFrom', [], [])], states

    elif root_node.type in def_statement:
        name = root_node.child_by_field_name('name')
        value = root_node.child_by_field_name('value')
        if value is None:
            indexs = tree_to_variable_index(name, index_to_code)
            for index in indexs:
                idx, code = index_to_code[index]
                pdg.append((code, idx, 'comesFrom', [], []))
                states[code] = [idx]
            return sorted(pdg, key=lambda x: x[1]), states
        else:
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            temp, states = PDG_ruby(value, index_to_code, states)
            pdg += temp
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in assignment:
        left_nodes = [x for x in root_node.child_by_field_name('left').children if x.type != ',']
        right_nodes = [x for x in root_node.child_by_field_name('right').children if x.type != ',']
        if len(right_nodes) != len(left_nodes):
            left_nodes = [root_node.child_by_field_name('left')]
            right_nodes = [root_node.child_by_field_name('right')]
        if len(left_nodes) == 0:
            left_nodes = [root_node.child_by_field_name('left')]
        if len(right_nodes) == 0:
            right_nodes = [root_node.child_by_field_name('right')]
        if root_node.type == "operator_assignment":
            left_nodes = [root_node.children[0]]
            right_nodes = [root_node.children[-1]]

        for node in right_nodes:
            temp, states = PDG_ruby(node, index_to_code, states)
            pdg += temp

        for left_node, right_node in zip(left_nodes, right_nodes):
            left_tokens_index = tree_to_variable_index(left_node, index_to_code)
            right_tokens_index = tree_to_variable_index(right_node, index_to_code)
            for token1_index in left_tokens_index:
                idx1, code1 = index_to_code[token1_index]
                for token2_index in right_tokens_index:
                    idx2, code2 = index_to_code[token2_index]
                    pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                states[code1] = [idx1]

        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in if_statement:
        pdg = []
        current_states = states.copy()
        others_states = []
        tag = False
        if 'else' in root_node.type:
            tag = True
        for child in root_node.children:
            if 'else' in child.type:
                tag = True
            if child.type not in if_statement:
                temp, current_states = PDG_ruby(child, index_to_code, current_states)
                pdg += temp
            else:
                temp, new_states = PDG_ruby(child, index_to_code, states)
                pdg += temp
                others_states.append(new_states)
        others_states.append(current_states)
        if not tag:
            others_states.append(states)
        new_states = {}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(pdg, key=lambda x: x[1]), new_states

    elif root_node.type in for_statement:
        pdg = []
        for i in range(2):
            left_nodes = [root_node.child_by_field_name('pattern')]
            right_nodes = [root_node.child_by_field_name('value')]
            assert len(right_nodes) == len(left_nodes)
            for node in right_nodes:
                temp, states = PDG_ruby(node, index_to_code, states)
                pdg += temp
            for left_node, right_node in zip(left_nodes, right_nodes):
                left_tokens_index = tree_to_variable_index(left_node, index_to_code)
                right_tokens_index = tree_to_variable_index(right_node, index_to_code)
                for token1_index in left_tokens_index:
                    idx1, code1 = index_to_code[token1_index]
                    for token2_index in right_tokens_index:
                        idx2, code2 = index_to_code[token2_index]
                        pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                    states[code1] = [idx1]
            temp, states = PDG_ruby(root_node.child_by_field_name('body'), index_to_code, states)
            pdg += temp
        # 添加控制依赖（循环体依赖循环条件）
        value = root_node.child_by_field_name('value')
        value_code = index_to_code[(value.start_point, value.end_point)][1]
        body = root_node.child_by_field_name('body')
        for body_child in body.children:
            body_code = index_to_code[(body_child.start_point, body_child.end_point)][1]
            pdg.append((body_code, value_code, 'controlDependsOn'))
        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in while_statement:
        pdg = []
        for i in range(2):
            for child in root_node.children:
                temp, states = PDG_ruby(child, index_to_code, states)
                pdg += temp
        # 添加控制依赖（循环体依赖循环条件）
        condition = root_node.child_by_field_name('condition')
        if condition:
            condition_code = index_to_code[(condition.start_point, condition.end_point)][1]
            body = root_node.child_by_field_name('body')
            for body_child in body.children:
                body_code = index_to_code[(body_child.start_point, body_child.end_point)][1]
                pdg.append((body_code, condition_code, 'controlDependsOn'))
        return sorted(pdg, key=lambda x: x[1]), states

    else:
        pdg = []
        for child in root_node.children:
            if child.type in do_first_statement:
                temp, states = PDG_ruby(child, index_to_code, states)
                pdg += temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp, states = PDG_ruby(child, index_to_code, states)
                pdg += temp
        return sorted(pdg, key=lambda x: x[1]), states

def PDG_go(root_node, index_to_code, states):
    assignment = ['assignment_statement']
    def_statement = ['var_spec']
    increment_statement = ['inc_statement']
    if_statement = ['if_statement', 'else']
    for_statement = ['for_statement']
    control_flow_nodes = ['if_statement', 'for_statement']

    states = states.copy()
    pdg = []

    def add_control_dependency(node, parent_node):
        """
        添加控制依赖关系
        """
        if parent_node:
            pdg.append((node, parent_node, 'controlDependsOn'))

    def add_data_dependency(node, parent_node):
        """
        添加数据依赖关系
        """
        if parent_node:
            pdg.append((node, parent_node, 'dataDependsOn'))

    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        idx, code = index_to_code[(root_node.start_point, root_node.end_point)]
        if root_node.type == code:
            return [], states
        elif code in states:
            return [(code, idx, 'comesFrom', [code], states[code].copy())], states
        else:
            if root_node.type == 'identifier':
                states[code] = [idx]
            return [(code, idx, 'comesFrom', [], [])], states

    elif root_node.type in def_statement:
        name = root_node.child_by_field_name('name')
        value = root_node.child_by_field_name('value')
        if value is None:
            indexs = tree_to_variable_index(name, index_to_code)
            for index in indexs:
                idx, code = index_to_code[index]
                pdg.append((code, idx, 'comesFrom', [], []))
                states[code] = [idx]
            return sorted(pdg, key=lambda x: x[1]), states
        else:
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            temp, states = PDG_go(value, index_to_code, states)
            pdg += temp
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in assignment:
        left_nodes = root_node.child_by_field_name('left')
        right_nodes = root_node.child_by_field_name('right')
        temp, states = PDG_go(right_nodes, index_to_code, states)
        pdg += temp
        name_indexs = tree_to_variable_index(left_nodes, index_to_code)
        value_indexs = tree_to_variable_index(right_nodes, index_to_code)
        for index1 in name_indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in value_indexs:
                idx2, code2 = index_to_code[index2]
                pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            states[code1] = [idx1]
        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in increment_statement:
        pdg = []
        indexs = tree_to_variable_index(root_node, index_to_code)
        for index1 in indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in indexs:
                idx2, code2 = index_to_code[index2]
                pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            states[code1] = [idx1]
        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in if_statement:
        pdg = []
        current_states = states.copy()
        others_states = []
        tag = False
        if 'else' in root_node.type:
            tag = True
        for child in root_node.children:
            if 'else' in child.type:
                tag = True
            if child.type not in if_statement and not tag:
                temp, current_states = PDG_go(child, index_to_code, current_states)
                pdg += temp
            else:
                temp, new_states = PDG_go(child, index_to_code, states)
                pdg += temp
                others_states.append(new_states)
        others_states.append(current_states)
        if not tag:
            others_states.append(states)
        new_states = {}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(pdg, key=lambda x: x[1]), new_states

    elif root_node.type in for_statement:
        pdg = []
        for child in root_node.children:
            temp, states = PDG_go(child, index_to_code, states)
            pdg += temp
        flag = False
        for child in root_node.children:
            if flag:
                temp, states = PDG_go(child, index_to_code, states)
                pdg += temp
            elif child.type == "for_clause":
                if child.child_by_field_name('update') is not None:
                    temp, states = PDG_go(child.child_by_field_name('update'), index_to_code, states)
                    pdg += temp
                flag = True
        # 添加控制依赖（循环体依赖循环条件）
        condition = root_node.child_by_field_name('condition')
        if condition:
            condition_code = index_to_code[(condition.start_point, condition.end_point)][1]
            body = root_node.child_by_field_name('body')
            for body_child in body.children:
                body_code = index_to_code[(body_child.start_point, body_child.end_point)][1]
                pdg.append((body_code, condition_code, 'controlDependsOn'))
        return sorted(pdg, key=lambda x: x[1]), states

    else:
        pdg = []
        for child in root_node.children:
            temp, states = PDG_go(child, index_to_code, states)
            pdg += temp
        return sorted(pdg, key=lambda x: x[1]), states

def PDG_php(root_node, index_to_code, states):
    """
    生成PHP代码的程序依赖图（PDG）
    """
    assignment = ['assignment_expression', 'augmented_assignment_expression']
    def_statement = ['simple_parameter']
    increment_statement = ['update_expression']
    if_statement = ['if_statement', 'else_clause']
    for_statement = ['for_statement']
    enhanced_for_statement = ['foreach_statement']
    while_statement = ['while_statement']
    do_first_statement = []
    control_flow_nodes = ['if_statement', 'for_statement', 'while_statement', 'foreach_statement']

    states = states.copy()
    pdg = []

    def add_control_dependency(node, parent_node):
        """
        添加控制依赖关系
        """
        if parent_node:
            pdg.append((node, parent_node, 'controlDependsOn'))

    def add_data_dependency(node, parent_node):
        """
        添加数据依赖关系
        """
        if parent_node:
            pdg.append((node, parent_node, 'dataDependsOn'))

    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        idx, code = index_to_code[(root_node.start_point, root_node.end_point)]
        if root_node.type == code:
            return [], states
        elif code in states:
            return [(code, idx, 'comesFrom', [code], states[code].copy())], states
        else:
            if root_node.type == 'identifier':
                states[code] = [idx]
            return [(code, idx, 'comesFrom', [], [])], states

    elif root_node.type in def_statement:
        name = root_node.child_by_field_name('name')
        value = root_node.child_by_field_name('default_value')
        if value is None:
            indexs = tree_to_variable_index(name, index_to_code)
            for index in indexs:
                idx, code = index_to_code[index]
                pdg.append((code, idx, 'comesFrom', [], []))
                states[code] = [idx]
            return sorted(pdg, key=lambda x: x[1]), states
        else:
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            temp, states = PDG_php(value, index_to_code, states)
            pdg += temp
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in assignment:
        left_nodes = root_node.child_by_field_name('left')
        right_nodes = root_node.child_by_field_name('right')
        temp, states = PDG_php(right_nodes, index_to_code, states)
        pdg += temp
        name_indexs = tree_to_variable_index(left_nodes, index_to_code)
        value_indexs = tree_to_variable_index(right_nodes, index_to_code)
        for index1 in name_indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in value_indexs:
                idx2, code2 = index_to_code[index2]
                pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            states[code1] = [idx1]
        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in increment_statement:
        pdg = []
        indexs = tree_to_variable_index(root_node, index_to_code)
        for index1 in indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in indexs:
                idx2, code2 = index_to_code[index2]
                pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            states[code1] = [idx1]
        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in if_statement:
        pdg = []
        current_states = states.copy()
        others_states = []
        tag = False
        if 'else' in root_node.type:
            tag = True
        for child in root_node.children:
            if 'else' in child.type:
                tag = True
            if child.type not in if_statement and not tag:
                temp, current_states = PDG_php(child, index_to_code, current_states)
                pdg += temp
            else:
                temp, new_states = PDG_php(child, index_to_code, states)
                pdg += temp
                others_states.append(new_states)
        others_states.append(current_states)
        if not tag:
            others_states.append(states)
        new_states = {}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(pdg, key=lambda x: x[1]), new_states

    elif root_node.type in for_statement:
        pdg = []
        for child in root_node.children:
            temp, states = PDG_php(child, index_to_code, states)
            pdg += temp
        flag = False
        for child in root_node.children:
            if flag:
                temp, states = PDG_php(child, index_to_code, states)
                pdg += temp
            elif child.type == "assignment_expression":
                flag = True
        # 添加控制依赖（循环体依赖循环条件）
        condition = root_node.child_by_field_name('condition')
        if condition:
            condition_code = index_to_code[(condition.start_point, condition.end_point)][1]
            body = root_node.child_by_field_name('statement')
            for body_child in body.children:
                body_code = index_to_code[(body_child.start_point, body_child.end_point)][1]
                pdg.append((body_code, condition_code, 'controlDependsOn'))
        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in enhanced_for_statement:
        name = None
        value = None
        for child in root_node.children:
            if child.type == 'variable_name' and value is None:
                value = child
            elif child.type == 'variable_name' and name is None:
                name = child
                break
        body = root_node.child_by_field_name('body')
        for i in range(2):
            temp, states = PDG_php(value, index_to_code, states)
            pdg += temp
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                states[code1] = [idx1]
            temp, states = PDG_php(body, index_to_code, states)
            pdg += temp
        # 添加控制依赖（循环体依赖循环条件）
        value_code = index_to_code[(value.start_point, value.end_point)][1]
        for body_child in body.children:
            body_code = index_to_code[(body_child.start_point, body_child.end_point)][1]
            pdg.append((body_code, value_code, 'controlDependsOn'))
        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in while_statement:
        pdg = []
        for i in range(2):
            for child in root_node.children:
                temp, states = PDG_php(child, index_to_code, states)
                pdg += temp
        # 添加控制依赖（循环体依赖循环条件）
        condition = root_node.child_by_field_name('condition')
        if condition:
            condition_code = index_to_code[(condition.start_point, condition.end_point)][1]
            body = root_node.child_by_field_name('statement')
            for body_child in body.children:
                body_code = index_to_code[(body_child.start_point, body_child.end_point)][1]
                pdg.append((body_code, condition_code, 'controlDependsOn'))
        return sorted(pdg, key=lambda x: x[1]), states

    else:
        pdg = []
        for child in root_node.children:
            temp, states = PDG_php(child, index_to_code, states)
            pdg += temp
        return sorted(pdg, key=lambda x: x[1]), states

def PDG_javascript(root_node, index_to_code, states):
    """
    生成JavaScript代码的程序依赖图（PDG）
    """
    assignment = ['assignment_pattern', 'augmented_assignment_expression']
    def_statement = ['variable_declarator']
    increment_statement = ['update_expression']
    if_statement = ['if_statement', 'else']
    for_statement = ['for_statement']
    while_statement = ['while_statement']
    do_first_statement = []
    control_flow_nodes = ['if_statement', 'for_statement', 'while_statement']

    states = states.copy()
    pdg = []

    def add_control_dependency(node, parent_node):
        """
        添加控制依赖关系
        """
        if parent_node:
            pdg.append((node, parent_node, 'controlDependsOn'))

    def add_data_dependency(node, parent_node):
        """
        添加数据依赖关系
        """
        if parent_node:
            pdg.append((node, parent_node, 'dataDependsOn'))

    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        idx, code = index_to_code[(root_node.start_point, root_node.end_point)]
        if root_node.type == code:
            return [], states
        elif code in states:
            return [(code, idx, 'comesFrom', [code], states[code].copy())], states
        else:
            if root_node.type == 'identifier':
                states[code] = [idx]
            return [(code, idx, 'comesFrom', [], [])], states

    elif root_node.type in def_statement:
        name = root_node.child_by_field_name('name')
        value = root_node.child_by_field_name('value')
        if value is None:
            indexs = tree_to_variable_index(name, index_to_code)
            for index in indexs:
                idx, code = index_to_code[index]
                pdg.append((code, idx, 'comesFrom', [], []))
                states[code] = [idx]
            return sorted(pdg, key=lambda x: x[1]), states
        else:
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            temp, states = PDG_javascript(value, index_to_code, states)
            pdg += temp
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in assignment:
        left_nodes = root_node.child_by_field_name('left')
        right_nodes = root_node.child_by_field_name('right')
        temp, states = PDG_javascript(right_nodes, index_to_code, states)
        pdg += temp
        name_indexs = tree_to_variable_index(left_nodes, index_to_code)
        value_indexs = tree_to_variable_index(right_nodes, index_to_code)
        for index1 in name_indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in value_indexs:
                idx2, code2 = index_to_code[index2]
                pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            states[code1] = [idx1]
        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in increment_statement:
        pdg = []
        indexs = tree_to_variable_index(root_node, index_to_code)
        for index1 in indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in indexs:
                idx2, code2 = index_to_code[index2]
                pdg.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            states[code1] = [idx1]
        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in if_statement:
        pdg = []
        current_states = states.copy()
        others_states = []
        tag = False
        if 'else' in root_node.type:
            tag = True
        for child in root_node.children:
            if 'else' in child.type:
                tag = True
            if child.type not in if_statement and not tag:
                temp, current_states = PDG_javascript(child, index_to_code, current_states)
                pdg += temp
            else:
                temp, new_states = PDG_javascript(child, index_to_code, states)
                pdg += temp
                others_states.append(new_states)
        others_states.append(current_states)
        if not tag:
            others_states.append(states)
        new_states = {}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(pdg, key=lambda x: x[1]), new_states

    elif root_node.type in for_statement:
        pdg = []
        for child in root_node.children:
            temp, states = PDG_javascript(child, index_to_code, states)
            pdg += temp
        flag = False
        for child in root_node.children:
            if flag:
                temp, states = PDG_javascript(child, index_to_code, states)
                pdg += temp
            elif child.type == "variable_declaration":
                flag = True
        # 添加控制依赖（循环体依赖循环条件）
        condition = root_node.child_by_field_name('condition')
        if condition:
            condition_code = index_to_code[(condition.start_point, condition.end_point)][1]
            body = root_node.child_by_field_name('statement')
            for body_child in body.children:
                body_code = index_to_code[(body_child.start_point, body_child.end_point)][1]
                pdg.append((body_code, condition_code, 'controlDependsOn'))
        return sorted(pdg, key=lambda x: x[1]), states

    elif root_node.type in while_statement:
        pdg = []
        for i in range(2):
            for child in root_node.children:
                temp, states = PDG_javascript(child, index_to_code, states)
                pdg += temp
        # 添加控制依赖（循环体依赖循环条件）
        condition = root_node.child_by_field_name('condition')
        if condition:
            condition_code = index_to_code[(condition.start_point, condition.end_point)][1]
            body = root_node.child_by_field_name('statement')
            for body_child in body.children:
                body_code = index_to_code[(body_child.start_point, body_child.end_point)][1]
                pdg.append((body_code, condition_code, 'controlDependsOn'))
        return sorted(pdg, key=lambda x: x[1]), states

    else:
        pdg = []
        for child in root_node.children:
            temp, states = PDG_javascript(child, index_to_code, states)
            pdg += temp
        return sorted(pdg, key=lambda x: x[1]), states

