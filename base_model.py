from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

llma_model_id = "meta-llama/Llama-3.2-3B-Instruct"
MAX_NEW_TOKENS = 256
MAX_LENGTH = 2048
MAX_TEMP = 0.7

class llm_generator():

    def __init__(self, model_id, hf_token):
        self.model_id = model_id
        self.hf_token = hf_token
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token = hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token = hf_token)
        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            return_full_text=False
        )
    
    def llm_generate(self, prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=MAX_TEMP):
        self.llm_response = self.generator(prompt, max_new_tokens=max_new_tokens, do_sample=True, 
                                           temperature=temperature)
        return self.llm_response
    
    def llm_encode(self, prompt):
        prompt_tokens = self.tokenizer.encode(prompt, truncation=True, max_length=MAX_LENGTH)
        return prompt_tokens
    
    def llm_decode(self, prompt_tokens):
        prompt_text = self.tokenizer.decode(prompt_tokens)
        return prompt_text