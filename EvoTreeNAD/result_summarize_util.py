import json
import os
import numpy as np
import re

def get_valid_node_info_filenames(folder):
    allnodeinfofilename = [xx for xx in os.listdir(folder) if 'all_node_info.json' in xx]
    r_files = [fname for fname in allnodeinfofilename if re.match(r"r\d+_all_node_info\.json", fname)]
    max_r = max([int(re.search(r"r(\d+)_all_node_info\.json", fname).group(1)) for fname in r_files]) if r_files else None
    print(max_r)
    max_r_filename = f"r{max_r}_all_node_info.json" if max_r is not None else None
    print(max_r_filename)
    if max_r_filename is None:
        return None, None
    
    print('Check empty nodes')
    with open(os.path.join(folder, max_r_filename), "r") as f:
        node_info_temp = json.load(f)
    print('Loaded all node info from:', max_r_filename)
    print(len(node_info_temp))

    node_info_temp_id_list = [int(xx['component_id'].split("model_")[1][:-3]) if ('model_' in xx['component_id']) else 0 for xx in node_info_temp]
    sorted_indices = np.argsort(node_info_temp_id_list)[-5:]
    node_info_temp_last5 = [node_info_temp[ii] for ii in sorted_indices]

    empty_node_count = 0
    for xx in node_info_temp_last5:
        if xx['reward'] <0:
            empty_node_count += 1

    print('the number of empty nodes in the last iteration is: ', empty_node_count)
    if empty_node_count >2:
        print("more than 2 empty nodes at the end")
        print("removing the last iteration")
        max_r -= 1
        max_r_filename = f"r{max_r}_all_node_info.json" if max_r >= 0 else None
        print(max_r_filename)
        print('Reload the node info file:')
    else:
        print("the last iteration is fine")
    return (max_r, max_r_filename)


def extract_node_info(node):
    # Extract component.id from rollout path
    rollout_component_ids = [
        n.component.id for n in node.rollout_path if hasattr(n, 'component')
    ]
    
    # Extract child component ids
    children_component_ids = [
        child.state.component.id for child in node.children
    ]

    info = {
        "component_id": node.state.component.id,
        "tokens_num": node.state.component.tokens_num,
        "reward": node.state.reward,
        "metrics": node.state.metrics,
        "best_score": node.state.best_score,
        "max_width": node.max_width,
        "ideas_list": node.ideas_list,
        "ideas_list_generated": node.ideas_list_generated,
        "ideas_list_used": node.ideas_list_used,
        "visits": node.visits,
        "value": node.value,
        "reward_list": node.reward_list,
        "rollout_component_ids": rollout_component_ids,
        "children_component_ids": children_component_ids,
        "outcome_str": node.state.outcome_str
    }
    return info
    
def traverse_tree_collect_info(root_node):
    all_info = []
    def dfs(node):
        # node.state.get_result()
        # node.state.get_component_outcome()
        all_info.append(extract_node_info(node))
        for child in node.children:
            dfs(child)
    dfs(root_node)
    return all_info

def print_tree_from_dict_to_string(nodeid2node, root_id=None, prefix=""):
    lines = []

    def recurse(node_id, prefix, is_last):
        node = nodeid2node[node_id]

        best_score = node.get("best_score", None)
        if best_score:
            tmp = round(best_score, 5)
        else:
            tmp = None

        connector = "└── " if is_last else "├── "
        line = prefix + connector + f"{node_id.split('/')[-1][:-3]}___{tmp}"
        lines.append(line)

        children = node.get("children_component_ids", [])
        for i, child_id in enumerate(children):
            is_last_child = (i == len(children) - 1)
            new_prefix = prefix + ("    " if is_last else "│   ")
            recurse(child_id, new_prefix, is_last_child)

    if root_id is None:
        all_ids = set(nodeid2node.keys())
        child_ids = {cid for node in nodeid2node.values() for cid in node.get("children_component_ids", [])}
        roots = list(all_ids - child_ids)
        if not roots:
            raise ValueError("Could not determine root node automatically.")
        root_id = roots[0]

    root_line = f"{root_id}[{nodeid2node[root_id].get('best_score', '')}]"
    lines.append(root_line)

    for i, child_id in enumerate(nodeid2node[root_id].get("children_component_ids", [])):
        recurse(child_id, "", i == len(nodeid2node[root_id]["children_component_ids"]) - 1)

    return "\n".join(lines)

import pickle

def save_all_node_info(root_node, modelcodegeneration_folder, save_prefixx = "", return_dict = False, idea_generator = None):
    all_node_info = traverse_tree_collect_info(root_node)
    
    nodeid2node = {}
    for xx in all_node_info:
        nodeid2node[xx['component_id']] = xx
    tree_node = print_tree_from_dict_to_string(nodeid2node)
    
    str000 = save_prefixx
    
    
    tempjsonpath = os.path.join(modelcodegeneration_folder, f"{str000}all_node_info.json")
    with open(tempjsonpath, 'w', encoding='utf-8') as file:
        json.dump(all_node_info, file, indent=4)
    
    tempjsonpath = os.path.join(modelcodegeneration_folder, f"{str000}node_tree.txt")
    with open(tempjsonpath, 'w', encoding='utf-8') as file:
        file.write(tree_node)

    if idea_generator:
        temppath= os.path.join(modelcodegeneration_folder, f"{str000}idea2embed_cache.pkl")
        try:
            idea2embed_cache = idea_generator.idea2embed_cache
        except:
            idea2embed_cache = {}
        with open(temppath, 'wb') as file:
            pickle.dump(idea2embed_cache, file)

    if return_dict:
        return({"all_node_info": all_node_info,  "node_tree": tree_node})
    



def get_node_value(node):
    return node.value

def get_node_reward(node):
    return(node.state.reward)

def get_node_score(node):
    return(node.state.best_score)
    
def get_node_id(node):
    return(node.state.component.id)


def find_all_parents(node):
    parents = []

    current = getattr(node, 'parent', None)
    while current is not None:
        node_info = {
            'id': get_node_id(current),
            'node': current,
            'value': get_node_value(current),
            'reward': get_node_reward(current),
            "score": get_node_score(current),
            
        }
        parents.append(node_info)
        current = getattr(current, 'parent', None)

    return parents


def find_best_node_best_reward_value_rm_leaf_forvalue(root_node):
    best_value_node_noleaf = None
    best_value_noleaf = float('-inf')

    best_value_node = None
    best_value = float('-inf')

    best_reward_node = root_node
    best_reward = get_node_reward(root_node)

    def dfs(node):
        nonlocal best_value_node_noleaf, best_value_noleaf
        nonlocal best_value_node, best_value
        nonlocal best_reward_node, best_reward

        value = get_node_value(node)
        reward = get_node_reward(node)

        # Best reward: include all nodes
        if reward > best_reward:
            best_reward = reward
            best_reward_node = node

        # Best value overall
        if value > best_value:
            best_value = value
            best_value_node = node

        # Best value (non-leaf only)
        if hasattr(node, 'children') and node.children:
            if value > best_value_noleaf:
                best_value_noleaf = value
                best_value_node_noleaf = node

        for child in getattr(node, 'children', []):
            dfs(child)

    dfs(root_node)

    return {
        "best_value_node_noleaf": best_value_node_noleaf,
        "best_value_node_id_noleaf": get_node_id(best_value_node_noleaf) if best_value_node_noleaf else None,
        "best_value_noleaf": best_value_noleaf,

        "best_value_node": best_value_node,
        "best_value_node_id": get_node_id(best_value_node) if best_value_node else None,
        "best_value": best_value,

        "best_reward_node": best_reward_node,
        "best_reward_node_id": get_node_id(best_reward_node) if best_reward_node else None,
        "best_reward": best_reward,
        
    }

def get_best_analysis_root_node(root_node):
    best_nodes_output = find_best_node_best_reward_value_rm_leaf_forvalue(root_node)
    allpara_best_reward_node = find_all_parents(best_nodes_output['best_reward_node'])
    all_para_id_best_rewards = [xx['id'] for xx in allpara_best_reward_node]
    best_value_node_id_noleaf_include_node_best_rewards = best_nodes_output['best_value_node_id_noleaf'] in all_para_id_best_rewards
    best_value_node_id_include_node_best_rewards = best_nodes_output['best_value_node_id'] in all_para_id_best_rewards
    print("best_value_node_id_noleaf_include_node_best_rewards: ", best_value_node_id_noleaf_include_node_best_rewards)
    print("best_value_node_id_include_node_best_rewards: ", best_value_node_id_include_node_best_rewards)
    
    best_reward_node = best_nodes_output['best_reward_node']
    print('-----reward----')
    print('best reward node id: ', best_nodes_output['best_reward_node_id'])
    print('its score:', get_node_score(best_reward_node))
    
    
    index_value = np.argmax([xx["value"] for xx in allpara_best_reward_node])
    node_best_value_among_parent_bestrewardnode = allpara_best_reward_node[index_value]
    return({"best_nodes_output": best_nodes_output, 
           "allpara_best_reward_node":allpara_best_reward_node, 
            "all_para_id_best_rewards":all_para_id_best_rewards, 
            "best_value_node_id_noleaf_include_node_best_rewards":best_value_node_id_noleaf_include_node_best_rewards, 
            "best_value_node_id_include_node_best_rewards":best_value_node_id_include_node_best_rewards, 
            "node_best_value_among_parent_bestrewardnode":node_best_value_among_parent_bestrewardnode, 
           })
