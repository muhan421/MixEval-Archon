# File: mix_eval/models/moa.py

import os
import asyncio
import time
from together import AsyncTogether, Together
from mix_eval.models.base_api import APIModelBase
from mix_eval.api.registry import register_model
import pdb

@register_model("moa")
class MoAModel(APIModelBase):
    def __init__(self, args):
        super().__init__(args)
        #pdb.set_trace()
        self.args = args
        self.client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        self.async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))
        self.reference_models = [
            "Qwen/Qwen2-72B-Instruct",
            "Qwen/Qwen1.5-72B-Chat",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
            "databricks/dbrx-instruct",
        ]
        self.aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"
        self.aggreagator_system_prompt = "...synthesize these responses into a single, high-quality response... Responses from models:"

    async def run_llm(self, model, prompt):
        response = await self.async_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=self.MAX_NEW_TOKENS,
        )
        return response.choices[0].message.content

    async def generate_moa_response(self, prompt):
        results = await asyncio.gather(*[self.run_llm(model, prompt) for model in self.reference_models])
        
        finalStream = self.client.chat.completions.create(
            model=self.aggregator_model,
            messages=[
                {"role": "system", "content": self.aggreagator_system_prompt},
                {"role": "user", "content": ",".join(str(element) for element in results)},
            ],
            stream=True,
        )

        response = ""
        for chunk in finalStream:
            response += chunk.choices[0].delta.content or ""
        
        return response

    def _decode(self, inputs):
        prompt = inputs[0]["content"]
        response = asyncio.run(self.generate_moa_response(prompt))
        time.sleep(self.FIX_INTERVAL_SECOND)
        return response

    def decode(self, inputs):
        delay = 1
        for i in range(self.MAX_RETRY_NUM):
            try:
                response_content = self._decode(inputs)
                print(f"final response_content: {response_content}")
                return response_content
            except Exception as e:
                print(f"Error in decode, retrying...")
                print(e)
                time.sleep(1)
                continue
        print(f"Failed after {self.MAX_RETRY_NUM} retries.")
        return 'Error'
