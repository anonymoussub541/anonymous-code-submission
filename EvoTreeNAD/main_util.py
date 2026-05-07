import importlib.util
import os
import re
import json
import sys
from prompt_upgrade_util import *
from base_util import *
from training_related_util import PROMPT_EXCEED_PARAS, PROMPT_SMALL_PARAS
import time
import math
import random
from collections import defaultdict, deque
from typing import Iterator, List
import torch




def get_node_depth(node):
    depth = 0
    cur = node
    while getattr(cur, "parent", None) is not None:
        depth += 1
        cur = cur.parent
    return depth


class ComponentIDGenerator:
    def __init__(self, basedir):
        self.basedir = basedir
        self.id_pool = []
        self.global_index = 1
    def generate_id(self, codestr):
        pathtmp = os.path.join(self.basedir,  f"model_{self.global_index}.py")
        dump_py_file(codestr, pathtmp, True)
        self.global_index += 1
        self.id_pool.append(pathtmp)
        return(pathtmp)



def convert_outcome_str(cid_outcome_list, supple_context = None):
    context = []
    jj = 1
    has_models = False
    for ii,kk in enumerate(cid_outcome_list, 1):
        content_py = load_py_file(kk['cid'])
        if len(content_py) < 100:
            continue
        context.append("--"*8)
        context.append(f"### **MODEL VERSION {jj}**:")
        context.append("```python")
        context.append(content_py)
        context.append("```")
        context.append(kk['evaluation_result'])
        context.append("")
        jj += 1
        has_models = True

    if has_models:
        context = ['## EXISTING MODEL VERSIONS & METRICS'] + context
    else:
        context = ["## No existing model versions. Propose new model designs from scratch."]
    
    if supple_context:
        context.append("")
        context.append("--"*8)
        context.append(supple_context)
    return("\n".join(context))


def count_trainable_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params




def average_top_percent(lst_input, perci = 10):
    # Validate percentage
    if not (0 <= perci <= 100):
        raise ValueError("Percentage must be between 0 and 100.")
    if perci == 0: # greedy
        if not lst_input:
            return 0
        return max(lst_input)
    lst = [x for x in lst_input if x >= 0]
    if not lst:
        return 0 
    sorted_lst = sorted(lst, reverse=True)
    top_count = max(1, int(len(sorted_lst) * perci / 100.0))
    if len(sorted_lst) >= 7 and top_count == 1:
        top_count = 2
        return(sorted_lst[0]*0.9 + sorted_lst[1]*0.1)
    top_percent = sorted_lst[:top_count]
    return sum(top_percent) / len(top_percent)

def random_generator(seed=None):
    rng = random.Random(seed)  # create a local Random instance with seed
    while True:
        yield rng.random()


class AdaptiveWidth:
    def __init__(self, top_k=5, WIDTH_D0 = 20, WIDTH_D1 = 8, WIDTH_best = 8, WIDTH_others = 5):
        self.modelid2reward = {}
        self.top_k = top_k
        self.WIDTH_D0 = WIDTH_D0
        self.WIDTH_D1 = WIDTH_D1
        self.WIDTH_best = WIDTH_best
        self.WIDTH_others = WIDTH_others

    def update_dict(self, model_id, reward):
        self.modelid2reward[model_id] = reward

    def get_best_model_id_top_k(self):
        # Sort model IDs by reward in descending order and return the top_k IDs
        sorted_models = sorted(self.modelid2reward.items(), key=lambda x: x[1], reverse=True)
        return {model_id:rw for model_id, rw in sorted_models[:self.top_k]}

    def get_adaptive_width(self, node):
        node_depth = get_node_depth(node)
        best_model_id_dict = self.get_best_model_id_top_k()
        if node_depth == 0:
            return self.WIDTH_D0
        elif node_depth == 1:
            return self.WIDTH_D1
        elif node.state.component.id in best_model_id_dict:
            return self.WIDTH_best
        else:
            return self.WIDTH_others



class Component:
    def __init__(self, cid, tokens_num = None):
        self.id = cid
        self.tokens_num = tokens_num


class Node:
    def __init__(self, state, parent=None, max_width = 0, upgrade_strategy = None, 
                 reward_engine = None,  evaluate_metric_engine = None, exploration_random_generator = None,):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.reward_list = []
        self.max_width = max_width
        self.rollout_path = []
        self.upgrade_strategy = upgrade_strategy
        self.reward_engine = reward_engine
        self.evaluate_metric_engine = evaluate_metric_engine
        self.ideas_list = None
        self.ideas_list_generated = []
        self.ideas_list_used = []
        self.ideas_list_used_and_result_pair = []
        self.fail_attempt_num = 0
        self.fail_attempt_num_limit = 6
        if exploration_random_generator is None:
            raise ValueError("Must specify exploration_random_generator!")
        self.exploration_random_generator = exploration_random_generator
        
    def is_fully_expanded(self):
        fully_expanded_flag = len(self.children) >= self.max_width
        return fully_expanded_flag

    def get_outcome_list(self, max_parent_len = 1, add_model_arch_str = False):
        evaluation_result_self = self.state.get_component_outcome()
        if add_model_arch_str:
            try:
                self_model_arch_str = self.state.metrics.get("model_arch_str", None)
            except:
                self_model_arch_str = None
            if self_model_arch_str:
                evaluation_result_self += f"\n\n -- Model Architecture String:\n{self_model_arch_str}\n\n"

        result_temp = [{"cid": self.state.component.id, "evaluation_result": evaluation_result_self, "best_score": self.state.best_score}]

        node_tmp = self
        for i in range(max_parent_len):
            node_pred = node_tmp.parent
            if node_pred is not None:
                result_temp.append({"cid": node_pred.state.component.id, "evaluation_result": node_pred.state.get_component_outcome(), "best_score": self.state.best_score})
                node_tmp = node_pred
            else:
                break
        return(result_temp[::-1]) 

    def select_child(self, add_noise = False, noise_range = 0.00075, nperci_top = 10):
        def clamp00(x, min_ = 0.001, max_ = 0.99):
            return(min(max_, max(min_,x)))
        
            
        def compute_weight(cmtr): ### here compute value
            reward0 = average_top_percent(cmtr['reward_list'], perci = nperci_top) # value
            explore0 = clamp00(0.8 - cmtr["visits"]/(self.visits+0.001)) # depressed, it is for traditional UCB, not suitable for EvoTreeNAD
            return([reward0, explore0])
               
        children_metrics = []
        for child in self.children:
            temp_dict = { "visits": child.visits, "value": child.value,
                        "reward": child.state.reward, "reward_list": child.reward_list}
            children_metrics.append(compute_weight(temp_dict))

        choices_weights = [xx[0] for xx in children_metrics]
        if add_noise: ## add noise on Jul 31 2025
            id000 = choices_weights.index(max(choices_weights))
            choices_weights = [xx+random.uniform(0, noise_range) for xx in choices_weights]
            id111 = choices_weights.index(max(choices_weights))
            if id000!=id111:
                print('!!add noise affects!')
            
        return self.children[choices_weights.index(max(choices_weights))]
        
    def backpropagate(self, result, nperci_top = 10):
        self.visits += 1
        self.reward_list.append(result) # store all rewards, important for computing value
        self.value = average_top_percent(self.reward_list, perci = nperci_top)
        if self.parent:
            self.parent.backpropagate(result, nperci_top)

    def expand_rollout(self, max_depth = 1, extend_max_depth = 2, max_parent_len = 1, max_parent_len_ideagen = 1, idea_generator = None, ideas_kk = 5,
    exclude_used_ideas = False, exclude_peer_ideas = False, exclude_parent_len = 0, ideasummary_agent = None,
    disruptor_k = 0, add_model_arch_str = False):
        depth = 0
        result_temp = self.get_outcome_list(max_parent_len,)
        parents_result_temp = result_temp[-(max_parent_len):]
        if max_parent_len <=0:
            parents_result_temp = result_temp[-1:]
        
        reward_final_list = []
        if max_depth < 1:
            raise ValueError("max_depth must >= 1!")

        if idea_generator:
            if not self.ideas_list:
                result_temp_ideagen = self.get_outcome_list(max_parent_len_ideagen, add_model_arch_str=add_model_arch_str)
                tokens_model = self.state.component.tokens_num
                print("tokens_model: ", tokens_model)
                addition_requirement_prompt = ""
                if tokens_model is not None:
                    if tokens_model > 3000:
                        print('trigger addition_requirement_prompt for idea generation')
                        addition_requirement_prompt = '\nAdditional requirement: Models appear to be code-heavy, potentially containing ineffective or overly complex components that could be simplified or removed without significantly impacting performance. Please ensure that at least 2 of the proposed upgrade ideas focus on subtractive upgrade, such as simplifying, optimizing, or removing ineffective components.\n'
                    else:
                        addition_requirement_prompt = "\nFocus on proposing refined/innovative and effective model upgrade ideas that can lead to significant performance improvements. That is, most ideas should be Exploratory and General upgrade.\n"
                
                if exclude_used_ideas and ideasummary_agent is not None:
                    parents_used_ideas = []
                    peer_used_ideas = []
                    if exclude_parent_len > 0:
                        node_temp = self
                        ii = 0
                        while ii < exclude_parent_len and node_temp.parent is not None:
                            if node_temp.parent.ideas_list_used:
                                parents_used_ideas += node_temp.parent.ideas_list_used
                            node_temp = node_temp.parent
                            ii += 1
                    if exclude_peer_ideas:
                        if self.parent is not None:
                            for peer_node in self.parent.children:
                                if peer_node is not self and peer_node.ideas_list_used:
                                    peer_used_ideas += peer_node.ideas_list_used
                    print('exclude_used_ideas: ', len(parents_used_ideas), len(peer_used_ideas))
                    content_idea_used_summary = ideasummary_agent.summarize_ideas(parents_used_ideas + peer_used_ideas)
                    summarized_ideas_str = content_idea_used_summary.get('summarized_ideas', None)
                    if summarized_ideas_str is not None:
                        temp_path = self.state.component.id.replace('.py', '_summarized_ideas_str.txt')
                        with open(temp_path, 'a') as f:
                            f.write(summarized_ideas_str)
                            f.write(f"=======\n\n")
                        addition_requirement_prompt += f"\n\n###Previously used ideas summary (DO NOT REPEAT OR OVERLAP! Propose ideas distinct with the previously used ideas):\n{summarized_ideas_str}\n"


                if self.ideas_list_generated:
                    addition_requirement_prompt += f"\n###Previously generated ideas (DO NOT REPEAT OR OVERLAP! Propose ideas distinct with the previously generated ideas):\n" + "\n".join([f"- {xx}" for xx in self.ideas_list_generated]) + "\n"

                model_version_str_input = convert_outcome_str(result_temp_ideagen)
                idea_agent_response = idea_generator.generate_ideas(model_version_str_input, ideas_kk, addition_requirement_prompt, disruptor_k = disruptor_k) # revised on oct 21, 2025
                self.ideas_list = idea_agent_response.get('ideas_generated', [])
                remark_temp = idea_agent_response.get('remark', None)
                if not isinstance(self.ideas_list, list):
                    self.ideas_list = []

                idea_save_path = self.state.component.id.replace('.py', '_ideas.txt')
                with open(idea_save_path, 'a') as f:
                    if remark_temp is not None:
                        f.write(f"### Remark from Idea Generation:\n{remark_temp}\n\n")
                    for idea in self.ideas_list[::-1]:
                        f.write(f"{idea}\n\n")
                    f.write(f"=======\n\n")
                print(idea_save_path)
                
                all_ideas_generated = idea_agent_response.get('all_ideas_generated', [])
                if all_ideas_generated is not None:
                    self.ideas_list_generated += all_ideas_generated #self.ideas_list
        used_idea = ""

        has_parent = self.parent is not None

        while depth < max_depth:
            cid_output_list_input = result_temp[-(max_parent_len+1):]
            print("cid_output_list_input: ", [xx['cid'].split('/')[-1] for xx in cid_output_list_input])
            if self.ideas_list:
                used_idea = self.ideas_list.pop()
                self.ideas_list_used.append(used_idea)
                supple_context_idea = f"\n### Model Upgrade Design Suggestion for next model version:\n{used_idea}\n"
                upgraded_component = self.upgrade_strategy.get_upgrade_component(cid_outcome_list = cid_output_list_input, supple_context = supple_context_idea)

            else:
                if has_parent:
                    upgraded_component = self.upgrade_strategy.get_upgrade_component(cid_output_list_input)
                if not has_parent:
                    upgraded_component = self.upgrade_strategy.get_upgrade_component_with_advance_llm(cid_outcome_list = cid_output_list_input)
            
            print('--depth: ', depth, '   used_idea: ', str(used_idea)[:100])
            
            new_state = ComponentState(upgraded_component, reward_engine = self.reward_engine, evaluate_metric_engine = self.evaluate_metric_engine)
            reward_final_list.append([new_state.get_result(), new_state])
            self.rollout_path.append(new_state)
            outcome_state = new_state.get_component_outcome()
            if "CUDA error: device-side assert triggered" in outcome_state:
                raise ValueError("CUDA error!")

            cid_state = new_state.component.id
            result_temp.append({"cid": cid_state, "evaluation_result": outcome_state})
            depth += 1
        reward_values = [xx[0] for xx in reward_final_list]
        
        training_excessive_runtime_flag = False
        if outcome_state:
            training_excessive_runtime_flag = "Training Halted - Excessive Runtime" in outcome_state
        
        if extend_max_depth > max_depth and ((max(reward_values) <= 0) or training_excessive_runtime_flag):
            trial_num = 0
            while depth < extend_max_depth:
                cid_output_list_input = parents_result_temp + result_temp[-1:]
                print("-Rollout: cid_output_list_input: ", [xx['cid'].split('/')[-1] for xx in cid_output_list_input])
                if used_idea:
                    supple_context0 = used_idea
                else:
                    supple_context0 = ""
                
                if trial_num > 2:
                    supple_context0 += f"\nNote: There are issues with the upgraded model code. **{trial_num} attempts** have failed!! The last model version shown above corresponds to the latest failed attempt with the error. The suggested upgrade idea may not work. Instead adopt the idea, you can adapt a similar idea or remove the defective component entirely.\n"
                elif trial_num > 0:
                    if "Timeout reached." in outcome_state:
                        print("===Timeout detected, adding prompt.")
                        supple_context0 += "\nProblem: The model initialization failed (might due to over-computation, inappropriate configuration, or inappropriate component design). Please remove, simplify, or find a safe alternate for components that might lead to failed initialization, over-computation, or excessive initialization time. \n"
                    else:
                        supple_context0 += f"\nNote: There are issues/errors with the upgraded model code. {trial_num} attempts have failed. The last model version shown above corresponds to the latest failed attempt with the error. Implement an effective, issue-specific modification to resolve it. If the issue is a bug, correct it, or otherwise remove the component causing the failure.\n"  ### revised on Aug 13, 2025                    

                upgraded_component = self.upgrade_strategy.get_upgrade_component(cid_output_list_input, supple_context = supple_context0, trial_num = trial_num)
                
                new_state = ComponentState(upgraded_component, reward_engine = self.reward_engine, evaluate_metric_engine = self.evaluate_metric_engine)
                reward_v_temp = new_state.get_result()
                reward_final_list.append([reward_v_temp, new_state])
                self.rollout_path.append(new_state)
                outcome_state = new_state.get_component_outcome()
                
                if "CUDA error: device-side assert triggered" in outcome_state:
                    raise ValueError("CUDA error!")

                cid_state = new_state.component.id
                result_temp.append({"cid": cid_state, "evaluation_result": outcome_state})
                depth += 1
                if reward_v_temp >= 0:
                    break
                trial_num += 1
            reward_values = [xx[0] for xx in reward_final_list]
        
        best_reward_values = max(reward_values)
        if best_reward_values <0:
            best_reward_state = reward_final_list[-1]
            if used_idea:
                print('delete used idea due to no best score')
                if used_idea in idea_generator.idea2embed_cache:
                    del idea_generator.idea2embed_cache[used_idea]
            self.fail_attempt_num += 1
            if self.fail_attempt_num <= self.fail_attempt_num_limit:
                return (None, None)                
        else:
            index00 = np.argmax(reward_values)
            best_reward_state = reward_final_list[index00]

        new_node = Node(best_reward_state[1], parent=self, reward_engine = self.reward_engine, 
                        evaluate_metric_engine = self.evaluate_metric_engine, exploration_random_generator = self.exploration_random_generator)
        self.children.append(new_node)  
        if used_idea:
            new_outcome = new_node.state.get_component_outcome()
            self.ideas_list_used_and_result_pair.append([used_idea, new_outcome])
        
        return (best_reward_state[0], new_node)
        

def EvoTree(root, iterations, rollout_depth=1, max_parent_len = 1, max_parent_len_ideagen = None, max_width=9, rollout_attempts = 2, idea_generator = None, ideas_kk = 0, upgrade_strategy = None, add_noise = False, noise_range = 0.0005, addnoise_bool_generator = None, tol_ept_nodes = 4, adaptive_width_obj = None, exclude_used_ideas = False, exclude_peer_ideas = False, exclude_parent_len = 0, ideasummary_agent = None, disruptor_k = 20, nperci_top = 10, add_model_arch_str = False): 
    if max_parent_len_ideagen is None:
        max_parent_len_ideagen = max_parent_len
    result_log00 = []

    for iii in range(iterations):
        add_noise_temp = add_noise
        if add_noise and addnoise_bool_generator:
            random_addnoise_flag = next(addnoise_bool_generator)
            if random_addnoise_flag:
                add_noise_temp = True
            else:
                print('disable add noise')
                add_noise_temp = False
            
        print('itr  ', iii)
        node = root
        while node.children and node.is_fully_expanded(): # find a un-full-fold node
            node = node.select_child( add_noise = add_noise_temp, noise_range = noise_range, nperci_top = nperci_top)

        if node.max_width == 0:
            if adaptive_width_obj is not None:
                node.max_width = adaptive_width_obj.get_adaptive_width(node)
            else:
                node.max_width = max_width

        node_depth = get_node_depth(node)
        if node_depth == 0:
            disruptor_k_used = 0
        else:
            disruptor_k_used = disruptor_k

        if node.upgrade_strategy is None:
            if upgrade_strategy is None:
                raise ValueError("Must specify upgrade_strategy!")
            node.upgrade_strategy = upgrade_strategy

        if ideas_kk == 0:
            ideas_kk_used = node.max_width
        else:
            if ideas_kk < node.max_width:
                ideas_kk_used = node.max_width
            else:
                ideas_kk_used = ideas_kk
        print('=ideas_kk_used: ', ideas_kk_used, '  node.max_width: ', node.max_width)
        result, new_node = node.expand_rollout(max_depth = rollout_depth, extend_max_depth = rollout_depth + rollout_attempts, max_parent_len = max_parent_len, max_parent_len_ideagen = max_parent_len_ideagen, idea_generator = idea_generator, ideas_kk = ideas_kk_used,
        exclude_used_ideas = exclude_used_ideas, exclude_peer_ideas = exclude_peer_ideas, exclude_parent_len = exclude_parent_len, ideasummary_agent = ideasummary_agent, disruptor_k = disruptor_k_used,
        add_model_arch_str = add_model_arch_str)
        if result is not None:
            new_node.backpropagate(result, nperci_top)
            if adaptive_width_obj is not None:
                adaptive_width_obj.update_dict(new_node.state.component.id, result)
            result_log00.append(result)
            if len(result_log00) >= tol_ept_nodes:
                if max(result_log00[-tol_ept_nodes:]) < 0:
                    raise ValueError(f"The last {tol_ept_nodes} nodes are empty!")
        print('TIMELOG:time each attempt: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))) 
    print(f"Finish {iterations} iterations!!")


import random
from typing import List, Iterator, Optional


class UpgradeComponentIdeaStrategy:
    """Strategy to upgrade components using model design ideas and LLM assistance."""
    def __init__(
        self,
        model_requirement_str: Optional[str] = None,
        multiple_llm: bool = False,
        llmagents=None,
        llm_indexer=None,
        component_id_generator=None,
        SYSTEM_PROMPT_UPGRADE_CODE: Optional[str] = None,
        refineagent = None,
        tau_trigger_refine: int = 2,
    ):
        if not (model_requirement_str or SYSTEM_PROMPT_UPGRADE_CODE):
            raise ValueError("Must specify either model_requirement_str or SYSTEM_PROMPT_UPGRADE_CODE")
        self.model_requirement_str = model_requirement_str
        self.multiple_llm = multiple_llm
        self.llmagents = llmagents
        self.llm_indexer = llm_indexer
        self.component_id_generator = component_id_generator
        
        if SYSTEM_PROMPT_UPGRADE_CODE is None:
            self.SYSTEM_PROMPT_UPGRADE_CODE = get_prompt_upgrade_code(model_requirement_str)
        else:
            self.SYSTEM_PROMPT_UPGRADE_CODE = SYSTEM_PROMPT_UPGRADE_CODE

        self.refineagent = refineagent
        self.tau_trigger_refine = tau_trigger_refine

    def get_upgrade_model_code(
        self,
        cid_outcome_list,
        supple_context: Optional[str] = None,
        trial_num: int = 0,
    ) -> str:
        if self.refineagent and trial_num >= self.tau_trigger_refine:
            print('===Trigger refine agent for upgrade code generation===')
            llmagent = self.refineagent
        else:
            llmagent = self.llmagents[next(self.llm_indexer)] if self.multiple_llm else self.llmagents   
        context_user_input = convert_outcome_str(cid_outcome_list, supple_context)
        
        msg_list_temp = llmagent.init_msg(context_user_input, self.SYSTEM_PROMPT_UPGRADE_CODE)
        try:
            response_temp = llmagent.request_llm_api_msgs(msg_list_temp)
            completion_tokens00 = response_temp[1]
            completion_tokens = None
            if completion_tokens00 is not None and isinstance(completion_tokens00, int):
                completion_tokens = completion_tokens00
            py_code = extract_python_code(response_temp[0])
        except Exception as e:
            print(f"Error during upgrade model code generation: {str(e)}")
            py_code = ""
            completion_tokens = 0
        return {'py_code': py_code, 'code_tokens_num': completion_tokens}

    def get_upgrade_component_with_advance_llm(self,
        cid_outcome_list,
        supple_context: Optional[str] = None,
    ):
        content_pure_dict = self.get_upgrade_model_code(cid_outcome_list, supple_context, trial_num = 100)
        content_pure = content_pure_dict['py_code']
        content_tokens_num = content_pure_dict['code_tokens_num']
        newcomp_id = self.component_id_generator.generate_id(content_pure)
        return Component(newcomp_id, content_tokens_num)

    def get_upgrade_component(
        self,
        cid_outcome_list,
        supple_context: Optional[str] = None,
        trial_num: int = 0,
    ):
        content_pure_dict = self.get_upgrade_model_code(cid_outcome_list, supple_context, trial_num)
        content_pure = content_pure_dict['py_code']
        content_tokens_num = content_pure_dict['code_tokens_num']
        newcomp_id = self.component_id_generator.generate_id(content_pure)
        return Component(newcomp_id, content_tokens_num)



from typing import List, Dict, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt

import numpy as np

from prompt_evaluation_util import POTENTIAL_GAIN_SYSTEM_PROMPT

class EvaluationMetricEngine:
    def __init__(self, train_main_customize_func, modelname, preprocessing_verify_func, init_paras_used, model_para_limit_B, model_para_extreme_low_threshold_B = 0, llmagent = None, evaluate_potential_gain: bool = False, use_worth_testing: bool = False, Potential_gain_scale: float = 0.1,
                 model_requirement_str: Optional[str] = None, Compliance_score_scale: float=0.05):
        self.train_main_customize_func = train_main_customize_func
        self.modelname = modelname
        self.preprocessing_verify_func = preprocessing_verify_func
        self.init_paras_used = init_paras_used
        self.model_para_limit_B = model_para_limit_B
        self.model_para_extreme_low_threshold_B = model_para_extreme_low_threshold_B
        self.llmagent = llmagent
        self.evaluate_potential_gain_flag = evaluate_potential_gain
        self.use_worth_testing = use_worth_testing
        self.worth_testing_threshold = 0.05
        if self.evaluate_potential_gain_flag and self.llmagent is None:
            raise ValueError("llmagent must be provided when evaluate_potential_gain is True.")
        
        self.model_requirement_str = model_requirement_str
        self.Compliance_score_scale = Compliance_score_scale
        
        self.Potential_gain_scale = Potential_gain_scale
        self.system_prompt = POTENTIAL_GAIN_SYSTEM_PROMPT
        if self.use_worth_testing:
            raise NotImplementedError("use_worth_testing is not implemented yet.")

        
    def __call__(self, component):
        result_metrics = self.evaluate_metrics(component, self.train_main_customize_func, self.modelname, self.preprocessing_verify_func, self.init_paras_used, model_para_limit_B = self.model_para_limit_B,
                                               model_para_extreme_low_threshold_B = self.model_para_extreme_low_threshold_B)
        return result_metrics
    
    def evaluate_potential_gain(self, component_code: str) -> Dict:
        msg_list_temp = self.llmagent.init_msg(component_code, POTENTIAL_GAIN_SYSTEM_PROMPT)
        i = 0
        potential_gain = 0.0
        worth_testing = None
        compliance_score = 0.0

        while True:
            i += 1
            response_temp = self.llmagent.request_llm_api_msgs(msg_list_temp)
            response_text = response_temp[0]
            try:
                # Extract JSON from the response
                json_start = response_text.index('{')
                json_end = response_text.rindex('}') + 1
                json_str = response_text[json_start:json_end]
                result_dict = json.loads(json_str)

                if "compliance_score" in result_dict:
                    if isinstance(result_dict["compliance_score"], (int, float)):
                        compliance_score = max(min(float(result_dict["compliance_score"]), 1.0), 0.0)
                result_dict["compliance_score"] = compliance_score

                if "potential_gain" in result_dict:
                    if isinstance(result_dict["potential_gain"], (int, float)):
                        result_dict["potential_gain"] = max(min(float(result_dict["potential_gain"]), 1.0), 0.0)
                        potential_gain = result_dict["potential_gain"]
                        if self.use_worth_testing:
                            try:
                                result_dict["worth_testing"] = max(min(float(result_dict["worth_testing"]), 1.0), 0.0)
                                worth_testing = result_dict["worth_testing"]
                            except Exception as e:
                                result_dict["worth_testing"] = None
                                print(f"Error parsing worth_testing value: {str(e)}")
                        return result_dict
            except Exception as e:
                print(f"Error parsing potential gain JSON for evaluation metrics: {str(e)}")
                pass
            if i > 3:
                break
        return {"potential_gain": potential_gain, "worth_testing": worth_testing, "compliance_score": compliance_score}

    def evaluate_metrics(self, component, eval_fun, modelname, preprocessing_verify, init_paras_used, model_para_limit_B = 1.0, model_para_extreme_low_threshold_B = 0.0):
        temp_module, error = import_specific_module_from_path(component.id)

        iflag = False
        default_metrics = [] ### must return a non-None metrics, default is []

        if temp_module is None:
            print(f"Import failed: {error}")
            # return({"metrics": default_metrics, "pass_check": False, "error_msg": f"Something wrong with the model code! {str(error)}"})
        else:
            if hasattr(temp_module, modelname):
                iflag = True
            else:
                print("The module does not contain the model!")
        if iflag:
            try:
                output_verify_temp = preprocessing_verify(component.id)
            except Exception as e:
                return({"metrics": default_metrics, "pass_check": False, "error_msg": "Error: The model failed the smoke test:\n"+str(e)})
                
            if output_verify_temp[0]:
                print("pass verification. next, will conduct evaluation.")
                try:
                    modelclass = getattr(temp_module, modelname)
                    if isinstance(init_paras_used, dict):
                        model0 = modelclass(**init_paras_used)
                    else:
                        model0 = modelclass(*init_paras_used)

                    try:
                        model_arch_str = str(model0)
                    except:
                        model_arch_str = None
                    model0_size = count_trainable_parameters(model0)/1e9
                    if model0_size > model_para_limit_B:
                        error_oom = PROMPT_EXCEED_PARAS
                        #"The model has too many parameters, yet its architecture is bulky, inefficient, and clearly in need of redesign! Rethink the architecture to provide a smart and sophisticated architecture!"
                        return({"metrics": default_metrics, "pass_check": True, "error_msg": error_oom, "model_size": model0_size, 'model_size_rw': 0.0, "model_arch_str": model_arch_str})
                    elif model0_size < model_para_extreme_low_threshold_B:
                        error_oom = PROMPT_SMALL_PARAS
                        return({"metrics": default_metrics, "pass_check": True, "error_msg": error_oom, "model_size": model0_size, 'model_size_rw': 0.0, "model_arch_str": model_arch_str})
                    
                    ## here evaluate models
                    potential_gain_value = 0.0
                    potential_gain_dict = {}
                    if self.evaluate_potential_gain_flag and self.use_worth_testing:
                        model_code = load_py_file(component.id)
                        if self.model_requirement_str is not None:
                            model_code = f"### Model Requirement:\n{self.model_requirement_str}\n\n### Model Code:\n" + model_code
                        potential_gain_dict = self.evaluate_potential_gain(model_code)
                        print("potential_gain_dict: ", potential_gain_dict)
                        potential_gain_value = potential_gain_dict.get("potential_gain", 0.0)
                        if self.use_worth_testing:
                            worth_testing_value = potential_gain_dict.get("worth_testing", None)
                            if worth_testing_value is not None:
                                if worth_testing_value < self.worth_testing_threshold:
                                    print("worth_testing is too low, skip evaluation.")
                                    return({"metrics": default_metrics, "pass_check": True, "error_msg": "The model architecture is not worth testing based on static analysis and diagnostics of model architecture. Please redesign the model architecture to improve its potential.", "model_size": model0_size, 'model_size_rw': 0.0, "remarks": str(potential_gain_dict),
                                            "model_arch_str": model_arch_str})

                    eval_metrics_dict = eval_fun(model0)
                    torch.cuda.empty_cache()

                    metrics_get = eval_metrics_dict.get('metrics', None)

                    if self.evaluate_potential_gain_flag and (not self.use_worth_testing) and metrics_get:
                        if len(metrics_get) > 2 and eval_metrics_dict.get('error_msg', None) is None:
                            model_code = load_py_file(component.id)
                            tmp = eval_metrics_dict.get('remark_info', '')
                            log00 = eval_metrics_dict.get('log', '')
                            supple = ""
                            if self.model_requirement_str is not None:
                                model_code = f"### Model Requirement:\n{self.model_requirement_str}\n\n### Model Code:\n" + model_code
                            potential_gain_dict = self.evaluate_potential_gain(model_code + supple)
                            print("potential_gain_dict: ", potential_gain_dict)
                            potential_gain_value = potential_gain_dict.get("potential_gain", 0.0)
                            compliance_score = potential_gain_dict.get("compliance_score", 0.0)
                            if 'aux_reward' in eval_metrics_dict: # added on Nov 26, 2025
                                eval_metrics_dict['aux_reward'] += self.Potential_gain_scale * potential_gain_value + compliance_score * self.Compliance_score_scale
                            else:
                                eval_metrics_dict['aux_reward'] = self.Potential_gain_scale * potential_gain_value + compliance_score * self.Compliance_score_scale
                        
                    eval_metrics_dict['model_size'] = model0_size
                    eval_metrics_dict['model_size_rw'] = 1.0 - model0_size/(model_para_limit_B+1e-4)
                    eval_metrics_dict["model_arch_str"] = model_arch_str
                    
                    if self.evaluate_potential_gain_flag:
                        eval_metrics_dict['potential_gain'] = potential_gain_value
                        eval_metrics_dict['potential_gain_dict'] = potential_gain_dict
                    return(eval_metrics_dict)
                except Exception as e:
                    print(f"fail verification. error: {str(e)}")
                    return({"metrics": default_metrics, "pass_check": True, "error_msg": str(e)})
            else:
                print("error: ", str(output_verify_temp[1]))
                return({"metrics": default_metrics, "pass_check": False, "error_msg": str(output_verify_temp[1])})
                
        print("fail verification.")
        return({"metrics": default_metrics, "pass_check": False, "error_msg": "Something wrong with the model code!"})
        
    
class EvaluationReward:
    def __init__(
        self,
        expect_metric_point: float,
        limit_metric: float,
        greater_is_better: bool,
        get_best_evaluation_metric: Callable[[List[Dict]], Optional[float]],
        process_evaluation_logs: Callable[[List[Dict]], List[Dict]] | None,
        model_size_reward: float = 0.0,
    ):
        print("Need to customize the config:")
        self.expect_metric_point = expect_metric_point
        self.limit_metric = limit_metric
        self.greater_is_better = greater_is_better
        self.k = 2.0 / abs(expect_metric_point - limit_metric)
        self.FAILED_REWARD = -0.001
        self.model_size_reward = model_size_reward
        if model_size_reward > 0.02 or model_size_reward < 0:
            raise ValueError("Please use 0~0.02 model_size_reward, the formula is model_size_reward * (1- model_size/limit_model_size). it is an additional reward. the reward limit of the model performance is 1.")

        if not callable(get_best_evaluation_metric):
            raise ValueError("get_best_evaluation_metric must be a callable function.")
        if process_evaluation_logs is not None:
            if not callable(process_evaluation_logs):
                raise ValueError("process_evaluation_logs must be a callable function.")

        self.get_best_evaluation_metric_fn = get_best_evaluation_metric
        self.process_evaluation_logs_fn = process_evaluation_logs

    def convert_to_rw(self, x: float) -> float:
        """Convert metric to reward using sigmoid-shaped curve."""
        if self.greater_is_better:
            return 1 / (1 + np.exp(-self.k * (x - self.expect_metric_point)))
        else:
            return 1 / (1 + np.exp(self.k * (x - self.expect_metric_point)))

    def select_metrics_display(self, metrics: List[Dict]) -> List[Dict]:
        if self.process_evaluation_logs_fn is None:
            return metrics
        return self.process_evaluation_logs_fn(metrics)

    def compute_reward(self, component_output_dict: Dict) -> Dict[str, Optional[float]]:
        metrics = component_output_dict.get("metrics", [])
        score_dict = self.get_best_evaluation_metric_fn(metrics)
        best_score = score_dict.get('best_score', None)
        final_dict = {"best_score": best_score}
        if 'score_for_reward' in score_dict:
            score_for_reward = score_dict.get('score_for_reward')
            final_dict["score_for_reward"] = score_for_reward
        else:
            score_for_reward = best_score
            
        reward = None
        if score_for_reward is not None and (not np.isnan(score_for_reward)):
            reward = self.convert_to_rw(score_for_reward)
            if np.isnan(reward):
                reward = None
            if 'aux_reward' in score_dict and reward is not None:
                final_dict["aux_reward"] = score_dict['aux_reward']
                reward += score_dict['aux_reward']
        if reward is None:
            reward = self.FAILED_REWARD
        else:
            if self.model_size_reward > 0:
                reward += self.model_size_reward * component_output_dict.get("model_size_rw", 0.0)
        final_dict["reward"] = reward
        
        return final_dict
        


import math
import random
from collections import defaultdict, deque


import random

class ComponentState:
    def __init__(self, component, reward_engine = None, evaluate_metric_engine = None):
        self.component = component
        self.metrics = None ## a dictionary with key: metrics, pass_check, error_msg
        self.reward = None
        self.best_score = None
        self.outcome_str = None
        self.reward_engine = reward_engine
        self.evaluate_metric_engine = evaluate_metric_engine

    def get_result(self):
        if self.metrics is None:
            self.metrics = self.evaluate_metric_engine(self.component)
            dump_py_file(json.dumps(self.metrics, indent = 4), convert_to_eval_output_path(self.component.id), True)
            
        if self.reward is None:
            reward_dict = self.reward_engine.compute_reward(self.metrics)
            try:
                self.reward = reward_dict["reward"]
            except:
                raise ValueError("No reward available!!! check reward_engine!!!")
            self.best_score = reward_dict.get("best_score", None)
            self.metrics['reward'] = self.reward
            self.metrics['best_score'] = self.best_score
            dump_py_file(json.dumps(self.metrics, indent = 4), convert_to_eval_output_path(self.component.id), True)
            
        return self.reward

    def get_component_outcome(self):
        if self.metrics is None:
            self.get_result()
        if self.outcome_str is None:
            metrics_temp = self.metrics.get("metrics", None)
            error_msg = self.metrics.get("error_msg", None)
            temp_str = "\n=====**Train Log and Evaluation Metrics**=====\n"
            if metrics_temp:
                metrics_temp_sel = self.reward_engine.select_metrics_display(metrics_temp)
                temp_str += json.dumps(metrics_temp_sel)

            if not self.metrics.get("pass_check", False):
                temp_str += "\n**Error:** The python code doesn't comply with the model requirements, please check, fix, and refine the code!"

            remark_info = self.metrics.get("remark_info", None)
            if remark_info:
                temp_str += f"\n{remark_info}\n"
                
            if error_msg:
                temp_str += f"\n**Detected: Bugs, training failure, or subpar model performance**:\n{error_msg}."
                
            self.outcome_str = temp_str
        return(self.outcome_str)

