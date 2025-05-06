
OPENAI_API_KEY=""
import os
import time
from openai import AzureOpenAI
import copy
class Chatgpt():
    def __init__(self):
        self.client = AzureOpenAI(
            api_key="",  
            api_version="2024-10-21",
            azure_endpoint="https://scy.openai.azure.com/"
        )

    def query(self,query,num_responses=1,temperature=0.3,top_p=0.7,max_tokens=2000):
        res=[]
        
        msg=copy.deepcopy(query)
        for i in range(num_responses):
            response= self.client.chat.completions.create(
                model="gpt-4o",
                messages=msg,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            res.append(response.choices[0].message.content)
            msg[-1]['content']=msg[-1]['content']
        time.sleep(0.2)
        return res