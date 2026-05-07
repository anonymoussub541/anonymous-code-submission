import os
from openai import AzureOpenAI, OpenAI
import requests
from dotenv import load_dotenv

load_dotenv()  

all_model_list = [ 'gpt4.1', 'gpt5chat', 'oss20', 'o3', ]


class AzureOpenAIChatClient:
    def __init__(self, modelname: str, max_completion_tokens = 10000, temperature = None, reasoning_effort= None, verbosity= None, direct_openai_flag = False):
        
        if reasoning_effort is not None and reasoning_effort not in ['low', 'medium', 'high']:
            print("reasoning_effort must be low, medium, or high!!!")
            print('set reasoning_effort to low')
            reasoning_effort = 'low'
        self.nonreasoning_models = ['gpt5chat', 'gpt4.1', 'oss20'] 
        if modelname in self.nonreasoning_models:
            reasoning_effort = None
        api_version = "2024-12-01-preview" 
        verbosity = None
        if modelname == 'o3':
            direct_openai_flag = True
        self.direct_openai_flag = direct_openai_flag

        if modelname == "gpt4.1":
            deployment = "gpt-4.1"
            api_version = "2025-01-01-preview"
        elif modelname == "gpt5chat":
            deployment = "gpt-5-chat"
        elif modelname == 'o3':
            deployment = "o3"
        elif modelname =='oss20':
            endpoint = os.getenv("OSS20_LOCAL_ENDPOINT")
            deployment = "/models/gpt-oss-20b-mxfp4.gguf"
            self.headers = {"Content-Type": "application/json"}
        else:
            raise ValueError("Invalid modelname!!!")
        
        if modelname == 'oss20':
            self.client = None
        elif self.direct_openai_flag:
            api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(
                api_key=api_key,
            )
            endpoint = None
        else:
            try:
                endpoint = os.getenv("AZURE_ENDPOINT")
                api_key = os.getenv("AZURE_OPENAI_API_KEY")
                self.client = AzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    api_key=api_key,
                )
            except Exception as e:
                print(f"Error initializing AzureOpenAI client: {e}; falling back to OpenAI client:")
                self.client = OpenAI(api_key=api_key)
                if deployment == 'gpt-5-chat':
                    deployment = "gpt-5-chat-latest"
                endpoint = None
        print(f'Initialized LLM Client with model {modelname}, deployment {deployment}')
        self.modelname = modelname
        self.deployment = deployment
        self.endpoint = endpoint
        setting_config = {'reasoning_effort': reasoning_effort, 'verbosity': verbosity, 'temperature': temperature, 'max_completion_tokens': max_completion_tokens}
        self.setting_config = {ii: vv for ii,vv in setting_config.items() if vv is not None}
        print(self.setting_config)
    
    def request_llm_api_msgs(self, messages):
        content = None
        completion_tokens = 0
        prompt_tokens = 0
        reasoning_tokens = 0
        completion_tokens_adjusted = 0
        try:
            if self.modelname == 'oss20':
                data = {'model': self.deployment, 'messages': messages,}
                data.update(self.setting_config)
                response = requests.post(self.endpoint, headers=self.headers, json=data)
                content_dict = response.json()
                content = content_dict['choices'][0]['message']['content']
                usage_dict = content_dict['usage']
                completion_tokens = usage_dict['completion_tokens']
                prompt_tokens = usage_dict['prompt_tokens']
                try:
                    reasoning_tokens = usage_dict['completion_tokens_details']['reasoning_tokens']
                except:
                    reasoning_tokens = 4000
                completion_tokens_adjusted = completion_tokens - reasoning_tokens         
                print(f'OSS20API----{completion_tokens}----{prompt_tokens}')
                return(content, completion_tokens_adjusted, prompt_tokens)
            else:
                response = self.client.chat.completions.create(
                        model = self.deployment,
                        messages = messages,
                        **self.setting_config,
                )
                try:
                    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
                except:
                    reasoning_tokens = 0
                completion_tokens = response.usage.completion_tokens
                completion_tokens_adjusted = completion_tokens - reasoning_tokens
                prompt_tokens = response.usage.prompt_tokens
                print(f'OPENAIAPI----{self.deployment}----{completion_tokens}----{prompt_tokens}----Reason:{reasoning_tokens}')
                content = response.choices[0].message.content
                return(content, completion_tokens_adjusted, prompt_tokens)
        except Exception as e:
            print(f"Error during API request: {e}")
            print(f'OPENAIAPI----{self.deployment}----{completion_tokens}----{prompt_tokens}----Reason:{reasoning_tokens}--Failed')
            return(content, completion_tokens_adjusted, prompt_tokens)

    def request_llm_api_prompts(self, query, system_prompt = None):
        msgs = self.init_msg(query, system_prompt)
        return(self.request_llm_api_msgs(msgs))
        
    def init_msg(self, query, system_prompt = None):
        if system_prompt is not None:
            return([{"role": "system", "content": system_prompt,}, {"role": "user", "content": query,}])
        else:
            return([{"role": "user", "content": query}])
            
    def update_msg(self, messages, response, query):
        messages += [{"role": "assistant", "content": response,}, {"role": "user", "content": query,}]
        return(messages)

