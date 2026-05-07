import os
import argparse
import yaml
import json
import numpy as np
import math
from typing import Dict

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default="0", help="CUDA device(s) to use, e.g., 0 or '0,1'")
parser.add_argument("--run_name", default="run1", help="run file name")
parser.add_argument("--resume", default=True, action=argparse.BooleanOptionalAction, help="whether resume from previous run")
parser.add_argument('--config_dir', type=str, default='./config', help='Path to the config directory, must contain config.yaml, model_requirements.py, customize_code.py')
parser.add_argument('--customized_run_config_path', type=str, default='./exp3/customized_run_config.json', help='Path to the customized run config json file')
parser.add_argument('--batch_iter_num', type=int, default=10, help='Maximum number of EvoTree iterations will be the value*10')


args = parser.parse_args()

print('Load config file!')
CONFIG_DIR = args.config_dir
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yaml")

total_batch_iter_num = args.batch_iter_num
print('total_batch_iter_num: ', total_batch_iter_num)

# check model_requirements.py config.yaml customize_code.py exists
for filename in ["model_requirements.py", "customize_code.py", "config.yaml"]:
    filepath = os.path.join(CONFIG_DIR, filename)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Required file '{filename}' not found in directory '{CONFIG_DIR}'.")

for filename in os.listdir(CONFIG_DIR):
    if filename.endswith(".py") and filename not in ["model_requirements.py", "customize_code.py", "verfun_final_new.py"]:
        raise ValueError(f"Unexpected Python file '{filename}' found in directory '{CONFIG_DIR}'. Do not include extra .py files in the config directory.")


model_requirements_py_path = os.path.join(CONFIG_DIR, "model_requirements.py")

import sys
sys.path.append(CONFIG_DIR)


try:
    with open(CONFIG_PATH, "r") as f:
        cfg_dict = yaml.safe_load(f)
    SAVEDIR = cfg_dict["project"]["SAVEDIR"]
    projectname = cfg_dict["project"]["name"]
    CODEDIR = cfg_dict["project"]["CODEDIR"]
except Exception as e:
    raise ValueError(f"Error loading config file: {e}")

import time

starting_time0 = time.time()

import sys
sys.path.append(CODEDIR)

# Set environment variables
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
print(f"Using CUDA devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
RUN_NAME = args.run_name
print("run name: ", RUN_NAME)
project_folder = os.path.join(SAVEDIR, projectname)
print(f"Save directory: {SAVEDIR}")
print(f"Project name: {projectname}")
print(f"Code directory: {CODEDIR}")

modelcodegeneration_folder = os.path.join(SAVEDIR, projectname, RUN_NAME)
os.makedirs(modelcodegeneration_folder, exist_ok=True)
print('all generated model will be saved in: ', modelcodegeneration_folder)

RESUME_EVOTREE = args.resume
print('RESUME_EVOTREE: ', RESUME_EVOTREE)


run_config_path = os.path.join(modelcodegeneration_folder, f'run_config.json')
loaded_run_config_flag = False

run_config = None

if RESUME_EVOTREE and os.path.isfile(run_config_path):
    print('Resume from previous run config!')
    try:
        with open(run_config_path, 'r') as f:
            run_config = json.load(f)
        loaded_run_config_flag = True
        print('Loaded run config')
    except Exception as e:
        raise ValueError(f"Error loading run config: {e}")


try:
    customized_run_config_path = args.customized_run_config_path
    with open(customized_run_config_path, 'r') as f:
        customized_run_config = json.load(f)
except:
    print('Error loading customized run config file!')
    if run_config is not None:
        customized_run_config = run_config
        print('Use loaded run config as customized run config!')
    else:
        raise ValueError('No customized run config file found and no loaded run config available!')



def deep_update(base, updates):
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base

if loaded_run_config_flag and run_config is not None:
    print('----use loaded run config to update the customized run config: ----')
    deep_update(customized_run_config, run_config)

print('----use customized run config: ----')
print(customized_run_config)



print('=====Load EvoTree parameters!=====')
LIMIT_B = customized_run_config["evotree"]["LIMIT_B"]
EXTREME_LOW_PARA_THRESHOLD_B = customized_run_config["evotree"]["EXTREME_LOW_PARA_THRESHOLD_B"]

TIMEOUT = customized_run_config["evotree"]["TIMEOUT"]
print('TIMEOUT: ',TIMEOUT)
Patience_Evolution = customized_run_config["evotree"].get("Patience_Evolution", 200)
TAU_coef_4earlystop = customized_run_config["evotree"]["TAU_coef_4earlystop"]
Overwrite_verfun_final_new_file = customized_run_config["reward"]["Overwrite_verfun_final_new_file"]

EXCLUDE_USED_IDEAS = customized_run_config["evotree"].get("exclude_used_ideas", False)

GENERATE_TASK_SPECIFIC_TRICK = customized_run_config["idea_generation"]["GENERATE_TASK_SPECIFIC_TRICK"]

LLM_API_NAME_SETTING_DICT = customized_run_config["llm_api_pool"]
APICALL_PROB_DICT_ideagen = customized_run_config["idea_generation"]["APICALL_PROB_DICT_ideagen"]
APICALL_PROB_DICT_modelupgrade = customized_run_config["model_upgrade"]["APICALL_PROB_DICT_modelupgrade"]
LLM_API_NAME_task_summary = customized_run_config["idea_generation"]["LLM_API_NAME_task_summary"]


IDEA_DIVERSITY_ENHANCE = False
if 'idea_diversity_enhance' in customized_run_config["idea_generation"]:
    IDEA_DIVERSITY_ENHANCE = customized_run_config["idea_generation"]['idea_diversity_enhance']  

initial_met_pair = customized_run_config.get("initial_met_pair", [])


print('Set up api agents!')
from llm_api_util import AzureOpenAIChatClient
### check availablity
if len(LLM_API_NAME_SETTING_DICT) < 1 or len(APICALL_PROB_DICT_modelupgrade) < 1 or len(APICALL_PROB_DICT_ideagen) < 1:
    raise ValueError("Error: LLM_API_NAME_SETTING_DICT or APICALL_PROB_DICT is empty!")
### check the key:
for kk in APICALL_PROB_DICT_modelupgrade:
    if kk not in LLM_API_NAME_SETTING_DICT:
        raise ValueError(f"Error: {kk} not in LLM_API_NAME_SETTING_DICT")
for kk in APICALL_PROB_DICT_ideagen:
    if kk not in LLM_API_NAME_SETTING_DICT:
        raise ValueError(f"Error: {kk} not in LLM_API_NAME_SETTING_DICT")


print('Run customize_code.py to load dataset and define train/eval function!')
import pandas as pd
import torch
from torch.utils.data import Dataset

from customize_code import *

from training_related_util import *
from base_util import *
from prompt_upgrade_util import *
from model_design_idea_prompt import *
from main_util import *
from result_summarize_util import *
from verify_runner_new import *
from collections import defaultdict
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score



train_loop_config = customized_run_config["train_loop_config"]


strict_early_stop = None
def train_main_customize(model):
    train_output = train_loop(
        model,
        evaluate= evaluate,
        train_loader= train_loader,
        test_loader = test_loader,
        per_device_train_batch_size = per_device_train_batch_size,
        per_device_eval_batch_size = per_device_eval_batch_size,
        batch_size = batch_size,  
        early_stop_fn = strict_early_stop,
        print_metrics= True,
        **train_loop_config,
    )
    return(train_output)
print('=====Finish all customized config!=====')













if not loaded_run_config_flag:
    # Initialize variables that would otherwise be undefined
    EXPECT_METRIC_POINT = globals().get('EXPECT_METRIC_POINT', 0.85)
    LIMIT_METRIC = globals().get('LIMIT_METRIC', 0.7)
    GREATER_IS_BETTER = globals().get('GREATER_IS_BETTER', True)
    per_device_train_batch_size = globals().get('per_device_train_batch_size', 32)
    per_device_eval_batch_size = globals().get('per_device_eval_batch_size', 32)
    batch_size = globals().get('batch_size', 32)
    task_specific_instruction = globals().get('task_specific_instruction', "")
    init_paras_used = globals().get('init_paras_used', {})
    
    knowledge_idea = ""

    knowledge_prompt = task_specific_instruction+'\n'+ knowledge_idea + '\n' + GENERAL_DESIGN_TRICK_KNOWLEDGE
    print('task_specific_instruction: ', task_specific_instruction)




if loaded_run_config_flag:
    EXPECT_METRIC_POINT = customized_run_config["data_config"]["EXPECT_METRIC_POINT"]
    LIMIT_METRIC = customized_run_config["data_config"]["LIMIT_METRIC"]
    GREATER_IS_BETTER = customized_run_config["data_config"]["GREATER_IS_BETTER"]
    per_device_train_batch_size = customized_run_config["data_config"]["per_device_train_batch_size"]
    per_device_eval_batch_size = customized_run_config["data_config"]["per_device_eval_batch_size"]
    batch_size = customized_run_config["data_config"]['batch_size']
    init_paras_used = customized_run_config['data_config']['init_paras_used']
    task_specific_instruction = customized_run_config['idea_generation']['task_specific_instruction']

# ========= finish config customize==========

print('Init API agents!')
tau4earlystop = max(EXPECT_METRIC_POINT, LIMIT_METRIC)* TAU_coef_4earlystop
print('EXPECT_METRIC_POINT, LIMIT_METRIC: ', EXPECT_METRIC_POINT, LIMIT_METRIC)
print('tau4earlystop: ', tau4earlystop)


def weighted_sampler(values, weights, seed=846241):
    # Normalize weights to sum to 1
    probs = np.array(weights, dtype=float)
    probs /= probs.sum()
    rng = np.random.default_rng(seed)
    while True:
        yield rng.choice(values, p=probs)


allllmagentspool = {ii: AzureOpenAIChatClient(**xx) for ii, xx in LLM_API_NAME_SETTING_DICT.items()}

def get_llm_agent_components(APICALL_PROB_DICT):
    if len(APICALL_PROB_DICT) > 1:
        llm_names = list(APICALL_PROB_DICT.keys())
        llm_call_prob = [APICALL_PROB_DICT[kk] for kk in llm_names]
        multiple_llm = True
        num_llm_agent = len(llm_names)
        llm_indexer = weighted_sampler(list(np.arange(num_llm_agent)), llm_call_prob)
        llm_agents = [allllmagentspool[kk] for kk in llm_names]
        print('use multiple llm: ', llm_names)
        print('with prob: ', llm_call_prob)
    else:
        llm_names = list(APICALL_PROB_DICT.keys())
        multiple_llm = False
        llm_indexer = None
        llm_agents = allllmagentspool[llm_names[0]]
        print('use single llm: ', llm_names)
    return multiple_llm, llm_indexer, llm_agents

def get_llm_agent_by_name(llm_name):
    if llm_name not in allllmagentspool:
        raise ValueError(f"llm_name {llm_name} not in allllmagentspool")
    llm_agent = allllmagentspool[llm_name]
    return llm_agent


multiple_llm_modelupgrade, llm_indexer_model_upgrade, llm_agent_modelupgrade = get_llm_agent_components(APICALL_PROB_DICT_modelupgrade)
multiple_llm_ideagen, llm_indexer_ideagen, llm_agent_ideagen = get_llm_agent_components(APICALL_PROB_DICT_ideagen)
llm_agent_task_summary = get_llm_agent_by_name(LLM_API_NAME_task_summary)
llm_agent_default = allllmagentspool[list(allllmagentspool.keys())[0]]



if EXCLUDE_USED_IDEAS:
    ideasummary_agent = IdeaSummaryAgent(llm_agent_task_summary)
else:
    ideasummary_agent = None


batch_path = os.path.join(project_folder, 'batch_test.pt')
try:
    if not os.path.isfile(batch_path):
        for ii, batch in enumerate(test_loader_small):
            torch.save(batch, batch_path)
            break
        print('save batch_test.pt for smoke testing of model!')
except Exception as e:
    print('Please check if test_loader_small is correctly set in your customize py file!')
    print('something wrong with create batch_test.pt')
    print(e)





print('Load model requirements!')
from model_requirements import *
modelname = model_requirements["model_name"]
model_requirement_str = load_py_file(model_requirements_py_path)

if customized_run_config["idea_generation"]["idea_generator_type"] == 'no_idea_agent':
    print('No idea generation agent will be used, skip task specific trick generation!')
    GENERATE_TASK_SPECIFIC_TRICK = False

print()
if GENERATE_TASK_SPECIFIC_TRICK and not loaded_run_config_flag:
    print('==Going to generate task specific trick:')
    refined_task_description, _, _ = get_refined_task_description(llm_agent_task_summary, str(model_requirements), task_specific_instruction)
    print(refined_task_description)
    
    refined_task_description_final = refined_task_description + '\n' + task_specific_instruction
    task_specific_trick_reference, _, _ = get_design_reference(llm_agent_task_summary, refined_task_description_final)

    print(task_specific_trick_reference)
    print('----')
    print('update the knowledge_idea, MODEL_DESIGN_SYSTEM_PROMPT_with_knowledge:')
    knowledge_idea = "\n\n".join([
            "\n\n-----\n## Quick‑Reference Inspiration Sheets (optional, knowledge about model architecture design)",
            task_specific_trick_reference,
        ])
    knowledge_prompt = refined_task_description_final+'\n'+ knowledge_idea + '\n' + GENERAL_DESIGN_TRICK_KNOWLEDGE
else:
    print('no task specifc trick need to be generated!')


if not loaded_run_config_flag:
    customized_run_config["data_config"]["key_metrics_name"] = key_metrics_name
    customized_run_config["data_config"]["EXPECT_METRIC_POINT"] = EXPECT_METRIC_POINT
    customized_run_config["data_config"]["LIMIT_METRIC"] = LIMIT_METRIC
    customized_run_config["data_config"]["GREATER_IS_BETTER"] = GREATER_IS_BETTER
    customized_run_config["data_config"]["per_device_train_batch_size"] = per_device_train_batch_size
    customized_run_config["data_config"]["per_device_eval_batch_size"] = per_device_eval_batch_size
    customized_run_config["data_config"]['batch_size'] = batch_size
    customized_run_config['data_config']['init_paras_used'] = init_paras_used
    customized_run_config['idea_generation']['task_specific_instruction'] = task_specific_instruction
    customized_run_config['idea_generation']['knowledge_prompt'] = knowledge_prompt
    with open(run_config_path, 'w') as f:
        json.dump(customized_run_config, f, indent=4)
else:
    knowledge_prompt = customized_run_config['idea_generation']['knowledge_prompt']
    print('load knowledge_prompt from run config!')



create_verfun_final_new_code = False
verfun_final_new_path = os.path.join(project_folder, "verfun_final_new.py")
if not os.path.isfile(verfun_final_new_path):
    create_verfun_final_new_code = True
    print('There is no verfun_final_new.py file, create one!')
elif Overwrite_verfun_final_new_file:
    create_verfun_final_new_code = True
    print('There is verfun_final_new.py file, will overwrite it!')
else:
    create_verfun_final_new_code = False
    print('There is verfun_final_new.py file, will use it!')

if create_verfun_final_new_code:
    code_str_temp = generate_verfun_final_new_code(modelname, batch_path, LIMIT_B, EXTREME_LOW_PARA_THRESHOLD_B, init_paras_used, verify_signature = False)
    dump_py_file(code_str_temp, os.path.join(project_folder, 'verfun_final_new.py'), force = True)


from verify_runner_new import preprocessing_verify as _preprocessing_verify
def preprocessing_verify(model_path: str):  # Changed default
    return _preprocessing_verify(model_path, TIMEOUT)


MODEL_SIZE_REWARD = customized_run_config["reward"]["MODEL_SIZE_REWARD"]

# Create instance (now enforced)
reward_engine = EvaluationReward(
    expect_metric_point=EXPECT_METRIC_POINT,
    limit_metric=LIMIT_METRIC,
    greater_is_better=GREATER_IS_BETTER,
    get_best_evaluation_metric=get_best_eval,
    process_evaluation_logs=process_eval_log,
    model_size_reward = MODEL_SIZE_REWARD,
)



def get_met_pair_for_early_stop(modelcodegeneration_folder, perc = 0.1, previous_pair = None):
    filelist = os.listdir(modelcodegeneration_folder)
    outputlist = []
    for xx in filelist:
        if 'evaloutput' in xx:
            outputlist.append(os.path.join(modelcodegeneration_folder, xx))
    print(len(outputlist))
    
    str00 = key_metrics_name
    path2metall = {}
    for path0 in outputlist:
        with open(path0, 'r') as f:
            temp = json.load(f)  # json.load() reads from file
        met = temp['metrics']
        if met:
            path2metall[path0] = [xx for xx in met if str00 in xx]# need to extract eval rows
    
    max00 = max([len(vv) for vv in path2metall.values()])
    print('max length of metrics', max00)
    metall = [vv for kk,vv in path2metall.items() if len(vv) == max00]
    print(len(metall))
    step00 = [yy['step'] for yy in metall[0]]
    print(step00)
    pdtmp = pd.DataFrame([[yy[str00] for yy in xx] for xx in metall])
    pdtmp.columns = step00
    pdtmp_sel = pdtmp.sort_values(step00[-1], ascending=(not GREATER_IS_BETTER)).iloc[:max(4, math.ceil(perc*len(pdtmp)))]
    
    met_pair = []
    tau0 = tau4earlystop 
    half_index = len(step00)//2
    step_index1 = int(len(step00)/3)
    step_index2 = int(2*len(step00)/3)
    if GREATER_IS_BETTER:
        for xx in step00[1: (max00-2)]:
            if xx <= step00[step_index1]:
                met_pair.append([xx, min(pdtmp_sel[xx]) - tau0])
            elif xx <= step00[step_index2]:
                met_pair.append([xx, min(pdtmp_sel[xx]) - tau0/2])
            else:
                met_pair.append([xx, min(pdtmp_sel[xx]) - tau0/4])
    else:
        for xx in step00[1: (max00-2)]:
            if xx <= step00[step_index1]:
                met_pair.append([xx, max(pdtmp_sel[xx]) + tau0])
            elif xx <= step00[step_index2]:
                met_pair.append([xx, max(pdtmp_sel[xx]) + tau0/2])
            else:
                met_pair.append([xx, max(pdtmp_sel[xx]) + tau0/4])

    print(met_pair)

    if previous_pair is not None:
        dict_temp00 = {}
        for ii,vv in met_pair:
            dict_temp00[ii] = vv
        for ii,vv in previous_pair:
            if ii in dict_temp00:
                if GREATER_IS_BETTER:
                    dict_temp00[ii] = max(vv, dict_temp00[ii])
                else:
                    dict_temp00[ii] = min(vv, dict_temp00[ii])
            else:
                dict_temp00[ii] = vv
        new_met_pair = [[kk, vv] for kk,vv in dict_temp00.items()]
        print('new met_pair: ', new_met_pair)
        return new_met_pair

    return(met_pair)



def get_strict_early_stop_fun(modelcodegeneration_folder, metpair = None, perc = 0.1, previous_pair = None):
    if metpair is None:
        METPAIR = get_met_pair_for_early_stop(modelcodegeneration_folder, perc, previous_pair)
    else:
        METPAIR = metpair
    def strict_early_stop(update_step: int, row: Dict) -> bool:
        if key_metrics_name in row:
            eval_acc = row[key_metrics_name]
            if GREATER_IS_BETTER:
                for step0, met0 in METPAIR:
                    if update_step == step0 and eval_acc < met0:
                        return True
            else:
                for step0, met0 in METPAIR:
                    if update_step == step0 and eval_acc > met0:
                        return True
        return False
    return(strict_early_stop)


EVALUATE_POTENTIAL_GAIN_FLAG = customized_run_config["reward"]["EVALUATE_POTENTIAL_GAIN_FLAG"]
USE_WORTH_TESTING_FLAG = customized_run_config["reward"]["USE_WORTH_TESTING_FLAG"]
if EVALUATE_POTENTIAL_GAIN_FLAG:
    LLM_API_NAME_EVALUATION_METRIC = customized_run_config["reward"]["LLM_API_NAME_EVALUATION_METRIC"]
    llm_agent_evaluation_metric = get_llm_agent_by_name(LLM_API_NAME_EVALUATION_METRIC)
else:
    llm_agent_evaluation_metric = None




evaluate_metric_engine = EvaluationMetricEngine(train_main_customize, modelname, preprocessing_verify, init_paras_used, LIMIT_B, model_para_extreme_low_threshold_B = EXTREME_LOW_PARA_THRESHOLD_B, llmagent = llm_agent_evaluation_metric, evaluate_potential_gain=EVALUATE_POTENTIAL_GAIN_FLAG, use_worth_testing = USE_WORTH_TESTING_FLAG, Potential_gain_scale=0.1, model_requirement_str=str(model_requirements), Compliance_score_scale=0.05)



component_id_generator = ComponentIDGenerator(modelcodegeneration_folder)

exploration_random_generator = random_generator(seed=652684)


use_disruptor_flag = False
if "use_disruptor_flag" in customized_run_config["idea_generation"]:
    use_disruptor_flag = customized_run_config["idea_generation"]["use_disruptor_flag"]
if use_disruptor_flag:
    from disruptor_ideas import IdeaAxesDisruptor
    import random
    seed0 = random.randint(0, 2**32 - 1)
    print('IdeaAxesDisruptor seed', seed0)
    ideaaxesdisruptor = IdeaAxesDisruptor(seed = seed0)
else:
    ideaaxesdisruptor = None




idea_generator_type = customized_run_config["idea_generation"]["idea_generator_type"]
if idea_generator_type == 'IdeaGenerationStrategy':
    analyze_prior_models = False
elif idea_generator_type == 'IdeaGenerationAgent_2phase':
    analyze_prior_models = True 
elif idea_generator_type == 'no_idea_agent':
    print('No idea generation agent will be used!')
else:
    raise ValueError(f'Unknown idea_generator_type: {idea_generator_type}')



USE_ADVANCED_MODEL_UPGRADE_encounter_multiplefailures = customized_run_config["model_upgrade"].get("USE_ADVANCED_MODEL_UPGRADE_encounter_multiplefailures", False)
LLM_API_NAME_advanced_model_upgrade = customized_run_config["model_upgrade"].get("LLM_API_NAME_advanced_model_upgrade", None)
if LLM_API_NAME_advanced_model_upgrade is None:
    LLM_API_NAME_advanced_model_upgrade = list(APICALL_PROB_DICT_modelupgrade.keys())[0]



### only used when idea_diversity_enhance is True
USE_MODEL_SIMILARITY_FOR_IDEAGEN = customized_run_config["idea_generation"].get("USE_MODEL_SIMILARITY_FOR_IDEAGEN", False)
MODEL_SIMILARITY_NAME = customized_run_config["idea_generation"].get("MODEL_SIMILARITY_NAME", None)
SCORE_TAU_IDEA_SIMILARITY = customized_run_config["idea_generation"].get("SCORE_TAU_IDEA_SIMILARITY", 0.8)
KK_IDEAS_PROHIBIT_SIMILARITY = customized_run_config["idea_generation"].get("KK_IDEAS_PROHIBIT_SIMILARITY", 5)

if not IDEA_DIVERSITY_ENHANCE:
    USE_MODEL_SIMILARITY_FOR_IDEAGEN = False

if USE_MODEL_SIMILARITY_FOR_IDEAGEN:
    try:
        print('Use model similarity for idea generation!')
        from sentence_transformers import SentenceTransformer
        model_similarity = SentenceTransformer(MODEL_SIMILARITY_NAME)
        model_similarity.cpu()
    except Exception as e:
        print('Failed to load model similarity model!')
        print(e)
        model_similarity = None
else:
    print('Not use model similarity for idea generation!')
    model_similarity = None


if USE_ADVANCED_MODEL_UPGRADE_encounter_multiplefailures:
    if LLM_API_NAME_advanced_model_upgrade not in LLM_API_NAME_SETTING_DICT:
        raise ValueError(f"API_NAME_ADVANCED_MODEL_UPGRADE {LLM_API_NAME_advanced_model_upgrade} not in LLM_API_NAME_SETTING_DICT")
    llm_agent_advanced_model_upgrade = allllmagentspool[LLM_API_NAME_advanced_model_upgrade]
    print('Use advanced model upgrade with llm agent: ', LLM_API_NAME_advanced_model_upgrade)
else:
    llm_agent_advanced_model_upgrade = None
    print('Not use advanced model upgrade!')


if idea_generator_type == 'no_idea_agent':
    print('No idea generation agent will be used!')
    idea_generator = None
else:
    idea_generator = IdeaGenerationStrategy(model_requirement_str = model_requirement_str, KNOWLEDGE_PROMPT = knowledge_prompt, multiple_llm = multiple_llm_ideagen, llmagents = llm_agent_ideagen, llm_indexer = llm_indexer_ideagen, disruptor = ideaaxesdisruptor, analyze_prior_models = analyze_prior_models, scores_tau = SCORE_TAU_IDEA_SIMILARITY, model_sim = model_similarity, kk_ideas_prohibit = KK_IDEAS_PROHIBIT_SIMILARITY, enhance_diversity = IDEA_DIVERSITY_ENHANCE)


TRIGGER_NUM_ADVANCED_MODEL_UPGRADE = customized_run_config["model_upgrade"].get("TRIGGER_NUM_ADVANCED_MODEL_UPGRADE", 5)

SYSTEM_PROMPT_UPGRADE_CODE_upgrade = get_prompt_upgrade_code(model_requirement_str)
upgrade_strategy = UpgradeComponentIdeaStrategy(model_requirement_str=model_requirement_str,
                            multiple_llm=multiple_llm_modelupgrade, llmagents= llm_agent_modelupgrade, llm_indexer= llm_indexer_model_upgrade, component_id_generator = component_id_generator,SYSTEM_PROMPT_UPGRADE_CODE=SYSTEM_PROMPT_UPGRADE_CODE_upgrade,
                            refineagent = llm_agent_advanced_model_upgrade, tau_trigger_refine = TRIGGER_NUM_ADVANCED_MODEL_UPGRADE,)



add_noise_prob = customized_run_config["evotree"]["add_noise_prob"]
import random
def bool_generator(seed=None):
    rng = random.Random(seed)
    while True:
        yield rng.random() < add_noise_prob 
addnoise_bool_generator = bool_generator(seed=542674)
print(f'add addnoise_bool_generator with {add_noise_prob} prob')


adaptive_width_obj = AdaptiveWidth(top_k=5, **customized_run_config["adaptivewidth_config"])



print('=======Begin EvoTree!=======')
print('EvoTree start from model0.py if exists, otherwise start from empty node!')
print(RESUME_EVOTREE)

if RESUME_EVOTREE:
    MAX_R, MAX_R_FILENAME = get_valid_node_info_filenames(modelcodegeneration_folder)
    print('max r is ', MAX_R)
    print('max r filename is ', MAX_R_FILENAME)
    if MAX_R_FILENAME is None:
        print('No file to resume! Start from model0 or empty node!')
        RESUME_EVOTREE = False


if not RESUME_EVOTREE:
    empty_node_flag = True
    model_path_start = os.path.join(modelcodegeneration_folder, 'model0.py')
    if os.path.isfile(model_path_start):
        empty_node_flag = False
        print('start node: ', model_path_start)

    if empty_node_flag:
        print("Start Node: Empty")
        model_path_start = os.path.join(modelcodegeneration_folder, 'empty.py')
        with open(model_path_start, 'w') as file:
            file.write('### No Previous Models Developed')
        start_comp = Component(model_path_start)
        start_state = ComponentState(start_comp, 
                                    reward_engine= reward_engine, 
                                    evaluate_metric_engine = evaluate_metric_engine)
        start_state.metrics = {"pass_check": True, "metrics": [], "error_msg": "Please craft a model architecture from scratch!"}
        start_state.reward = start_state.reward_engine.FAILED_REWARD
    else:
        start_comp = Component(model_path_start)
        start_state = ComponentState(start_comp, 
                                    reward_engine= reward_engine, 
                                    evaluate_metric_engine = evaluate_metric_engine)
    root_node = Node(start_state, max_width=0, 
                    upgrade_strategy = upgrade_strategy, 
                    reward_engine = reward_engine, 
                    evaluate_metric_engine = evaluate_metric_engine,
                    exploration_random_generator = exploration_random_generator,)
    current_met_pair = None
    best_model_idx = 0
    best_model_log = []
    start_iter_batch_id = 0

    if initial_met_pair:
        current_met_pair = initial_met_pair
    strict_early_stop = get_strict_early_stop_fun(modelcodegeneration_folder, metpair = current_met_pair)

else:
    print('RESUME EVOTREE!')
    print(modelcodegeneration_folder)
    print(MAX_R_FILENAME)
    print('Resume from max r node!')
    print('Load root node from max r node!')
    with open(os.path.join(modelcodegeneration_folder, MAX_R_FILENAME), 'r') as file:
        node_info_temp = json.load(file)

    import pickle
    temppath = os.path.join(modelcodegeneration_folder, f"r{MAX_R}_idea2embed_cache.pkl")
    if os.path.isfile(temppath):
        with open(temppath, 'rb') as handle:
            idea2embed_cache_loaded = pickle.load(handle)
        idea_generator.idea2embed_cache = idea2embed_cache_loaded
        print('load idea2embed_cache from ', temppath)
    else:
        print('no idea2embed_cache file found!')

    reward_temp = []
    all_model_id = set([])
    for xx in node_info_temp:
        reward_temp.append(xx['reward'])
        if 'empty.py' in xx['component_id'] or 'model0.py' in xx['component_id']:
            continue
        all_model_id.add(xx['component_id'].split("model_")[1][:-3])
        for yy in xx['rollout_component_ids']:
            all_model_id.add(yy.split("model_")[1][:-3])
        for yy in xx['children_component_ids']:
            all_model_id.add(yy.split("model_")[1][:-3])

    max_temp_id = np.argmax(reward_temp)
    best_reward_node_temp = node_info_temp[max_temp_id]
    best_model_idx = int(best_reward_node_temp['component_id'].split("model_")[1][:-3]) if ('model_' in best_reward_node_temp['component_id']) else 0
    print('best model idx is ', best_model_idx)
    existing_model_id_max = max([int(xx) for xx in all_model_id])
    print('the best reward in the last iteration is: ', best_reward_node_temp['reward'])
    print('best score is ', best_reward_node_temp['best_score'])
    print('model pool size: ', len(all_model_id))
    print('the max model id is: ', existing_model_id_max)

    component_id_generator.global_index = existing_model_id_max+1
    print(component_id_generator.global_index)

    node_dict = {}
    for xx in node_info_temp:
        component_id = xx['component_id']
        component_temp = Component(component_id, tokens_num = xx.get('tokens_num', None))
        state_temp = ComponentState(component_temp, reward_engine= reward_engine, 
                                    evaluate_metric_engine = evaluate_metric_engine)
        state_temp.reward = xx['reward']
        state_temp.best_score = xx['best_score']
        state_temp.outcome_str = xx['outcome_str']
        state_temp.metrics = xx['metrics']

        node_item_temp = Node(state_temp, parent=None, max_width = xx.get('max_width', 0),
                            upgrade_strategy = upgrade_strategy, reward_engine = reward_engine, 
                            evaluate_metric_engine = evaluate_metric_engine,
                            exploration_random_generator = exploration_random_generator)
        node_item_temp.visits = xx['visits']
        node_item_temp.value = xx['value']
        node_item_temp.reward_list = xx['reward_list']
        node_item_temp.ideas_list = xx.get('ideas_list', None)
        node_item_temp.ideas_list_used = xx.get('ideas_list_used', [])
        node_item_temp.ideas_list_generated = xx.get('ideas_list_generated', [])
        node_dict[component_id] = node_item_temp

    for xx in node_info_temp:
        current_component_id = xx['component_id']
        children_ids = xx['children_component_ids']
        if len(children_ids) ==0:
            continue

        node_dict[current_component_id].children = [node_dict[child_id] for child_id in children_ids if child_id in node_dict]
        for child in node_dict[current_component_id].children:
            child.parent = node_dict[current_component_id]

    
    find_root_node_flag = False
    for xx in node_dict:
        if 'empty.py' in xx:
            root_node = node_dict[xx]
            find_root_node_flag = True
            break
        elif 'model0.py' in xx:
            root_node = node_dict[xx]
            find_root_node_flag = True
            break
    print("find_root_node_flag: ", find_root_node_flag)
    if not find_root_node_flag:
        raise ValueError("Cannot find root node!")
    
    current_met_pair = None
    best_model_log = []
    start_iter_batch_id = MAX_R + 1

    if initial_met_pair:
        current_met_pair = initial_met_pair
    next_met_pair = get_met_pair_for_early_stop(modelcodegeneration_folder, perc = 0.1, previous_pair = current_met_pair)
    strict_early_stop = get_strict_early_stop_fun(modelcodegeneration_folder, metpair = next_met_pair)
    current_met_pair = next_met_pair



evotree_cols = ['rollout_depth', 'max_parent_len', 'max_parent_len_ideagen', 'rollout_attempts', 'add_noise', 'noise_range', 'tol_ept_nodes', 'exclude_used_ideas', 'exclude_peer_ideas', 'exclude_parent_len', 'nperci_top', 'add_model_arch_str']
evotree_config = {kk: customized_run_config["evotree"][kk] for kk in evotree_cols if kk in customized_run_config["evotree"]}
evotree_config['disruptor_k'] = customized_run_config['idea_generation'].get('disruptor_k', 0)

NPERCITOP = evotree_config.get('nperci_top', 10)
iters_per_batch = customized_run_config['evotree'].get('iters_per_batch', 10)

starting_time = time.time()
print('TIMELOG:EvoTree start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(starting_time))) ## print date time
print('TIMELOG:prepare time used: ', starting_time - starting_time0)


while start_iter_batch_id < total_batch_iter_num:
    if component_id_generator.global_index - best_model_idx > Patience_Evolution:
        print('best model id is ', best_model_idx)
        print('model pool max id is ', component_id_generator.global_index)
        print('break! finish the iteration!')
        break
    print('Iteration Batch -------', start_iter_batch_id, '------')


    addnoise_bool_generator_used = addnoise_bool_generator

    EvoTree(root_node, iterations = iters_per_batch,  idea_generator= idea_generator, upgrade_strategy= upgrade_strategy,
        addnoise_bool_generator = addnoise_bool_generator_used,
        adaptive_width_obj = adaptive_width_obj,
        ideasummary_agent = ideasummary_agent, **evotree_config)
    
    print('TIMELOG:time used for iterations so far: ', time.time() - starting_time)
    print('TIMELOG:time at iteractions ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))) 
    print(component_id_generator.global_index)
    node_summary_temp = save_all_node_info(root_node, modelcodegeneration_folder, save_prefixx = f"r{start_iter_batch_id}_", return_dict = True, idea_generator = idea_generator)

    all_node_statistic_list = []
    node_info_temp = node_summary_temp['all_node_info']
    for temp in node_info_temp:
        temp_dict_node = {"id": temp['component_id'].split('/')[-1],
                        "reward": temp['reward'],
                        "best_score": temp['best_score'],
                        "visit": temp['visits'],
                        'modelsize': temp['metrics'].get('remark_info', "")[14:-13]}
        all_node_statistic_list.append(temp_dict_node)
    all_node_statistic_df = pd.DataFrame(all_node_statistic_list)

    all_node_statistic_df_sorted = all_node_statistic_df.sort_values(by='reward', ascending=False)
    all_node_statistic_df_sorted.to_csv(os.path.join(modelcodegeneration_folder, f'r{start_iter_batch_id}_all_node_statistics_sorted.csv'), index = False)
    print('all node statistics saved!')



    main_trajaectory = []
    temp_node = root_node
    while temp_node.children:
        temp_node = temp_node.select_child(nperci_top = NPERCITOP)
        
        temp_dict_node = {"id": temp_node.state.component.id.split('/')[-1],
                        "reward": temp_node.state.reward,
                        "best_score": temp_node.state.best_score,
                        "visit": temp_node.visits,
                        'modelsize': temp_node.state.metrics.get('remark_info', "")[14:-13],
                        }
        main_trajaectory.append(temp_dict_node)
    main_trajaectory_df = pd.DataFrame(main_trajaectory).iloc[:-1]
    main_trajaectory_df.to_csv(os.path.join(modelcodegeneration_folder, f'r{start_iter_batch_id}_principle_lineage.csv'), index=False) # principle lineage
    print('main trajectory saved!')


    print(component_id_generator.global_index)
    analysis_result_temp = get_best_analysis_root_node(root_node)
    next_node1 = analysis_result_temp['best_nodes_output']['best_reward_node']
    print(get_node_score(next_node1))
    best_node_id = get_node_id(next_node1)
    print(best_node_id)
    cur_best_model_id = int(best_node_id[:-3].split('model_')[-1])
    if cur_best_model_id > best_model_idx:
        best_model_idx = cur_best_model_id
        print('new best model id: ', best_model_idx)

    print(adaptive_width_obj.get_best_model_id_top_k())

    analysis_result_temp = get_best_analysis_root_node(root_node)
    node_temp_best_reward = analysis_result_temp['best_nodes_output']['best_reward_node']
    best_node_id = get_node_id(node_temp_best_reward)
    best_node_score = get_node_score(node_temp_best_reward)
    depth_temp_best_reward = get_node_depth(node_temp_best_reward)
    print('best_node:', best_node_id, ' reward: ', best_node_score, 'depth: ', depth_temp_best_reward)
    best_model_log.append([depth_temp_best_reward, best_node_score, best_node_id])
    
    print('Update strict early stop function!')
    next_met_pair = get_met_pair_for_early_stop(modelcodegeneration_folder, perc = 0.1, previous_pair = current_met_pair)
    strict_early_stop = get_strict_early_stop_fun(modelcodegeneration_folder, metpair = next_met_pair)
    current_met_pair = next_met_pair
    start_iter_batch_id += 1
    
print('EvoTree finished! Reached total iterations or early stopping criteria met.')
