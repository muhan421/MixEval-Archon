from mix_eval.api.registry import register_model
import os
import asyncio
import time
from together import AsyncTogether, Together
from mix_eval.models.base_api import APIModelBase
from mix_eval.api.registry import register_model
import pdb

@register_model("dbrx_instruct_together")
class DBRX_Instruct_Together(APIModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        self.async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))
        self.model_name = "databricks/dbrx-instruct"

    async def run_llm(self, model, prompt):
        response = await self.async_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=self.MAX_NEW_TOKENS,
        )
        return response.choices[0].message.content

    def _decode(self, inputs):
        prompt = inputs[0]["content"]
        response = asyncio.run(self.run_llm(self.model_name, prompt))
        time.sleep(self.FIX_INTERVAL_SECOND)
        return response

    def decode(self, inputs):
        delay = 1
        for i in range(self.MAX_RETRY_NUM):
            try:
                response_content = self._decode(inputs)
                # print(f"final response_content: {response_content}")
                return response_content
            except Exception as e:
                print(f"Error in decode, retrying...")
                print(e)
                time.sleep(1)
                continue
        print(f"Failed after {self.MAX_RETRY_NUM} retries.")
        return 'Error'


    