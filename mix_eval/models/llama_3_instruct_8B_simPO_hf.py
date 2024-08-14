from mix_eval.api.registry import register_model
import os
import asyncio
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from mix_eval.models.base_api import APIModelBase
import pdb

@register_model("llama_3_instruct_8B_simPO_hf")
class Llama_3_Instruct_8B_SimPO_Hf(APIModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_name = "princeton-nlp/Llama-3-Instruct-8B-SimPO"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")

    async def run_llm(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.MAX_NEW_TOKENS,
            temperature=0.7,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _decode(self, inputs):
        prompt = inputs[0]["content"]
        response = asyncio.run(self.run_llm(prompt))
        time.sleep(self.FIX_INTERVAL_SECOND)
        return response

    def decode(self, inputs):
        delay = 1
        for i in range(self.MAX_RETRY_NUM):
            try:
                response_content = self._decode(inputs)
                return response_content
            except Exception as e:
                print(f"Error in decode, retrying...")
                print(e)
                time.sleep(1)
                continue
        print(f"Failed after {self.MAX_RETRY_NUM} retries.")
        return 'Error'