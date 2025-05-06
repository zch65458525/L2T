from transformers import AutoModelForCausalLM, AutoTokenizer,set_seed,pipeline
import random
import bitsandbytes
import numpy as np
import torch
class LLM():
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-72B-Instruct"
        self.path=""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            torch_dtype="auto",
            device_map="auto",
            load_in_4bit=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model.eval()
        torch.no_grad()

    def query(self, query,num_responses=1,temperature=0.7,top_p=0.9,max_tokens=512):
        
        sequences = []
        messages = [
        {"role": "system", "content": "You are a helpful assistant. Always follow the intstructions precisely and output the response exactly in the requested format."},
        ]
        for i in query:
            messages.append(i)
        for _ in range(num_responses):
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            random_int = random.randint(0, 1919810)
            set_seed(random_int)
            np.random.seed(random_int)
            torch.manual_seed(random_int)
            torch.cuda.manual_seed_all(random_int) 
            print(random_int)
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                top_k=50,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            sequences.append(response)
        print(sequences)
        return sequences
