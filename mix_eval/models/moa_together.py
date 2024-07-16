import os
import asyncio
import time
import copy
from together import AsyncTogether, Together
from mix_eval.models.base_api import APIModelBase
from mix_eval.api.registry import register_model
import pdb

def inject_references_to_messages(messages, references):
    messages = copy.deepcopy(messages)

    #pdb.set_trace()

    system = (
        "You have been provided with a set of responses from various open-source models to the latest user query. "
        "Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically "
        "evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. "
        "Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive "
        "reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of "
        "accuracy and reliability.\n\nResponses from models:"
    )

    for i, reference in enumerate(references):
        system += f"\n{i+1}. {reference}"

    if messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + system
    else:
        messages = [{"role": "system", "content": system}] + messages
	
    #pdb.set_trace()
    return messages

async def async_generate_fn(client, model, messages, temperature, max_tokens):
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

async def generate_with_references(
    client,
    model,
    messages,
    references=[],
    max_tokens=2048,
    temperature=0.7
):
    ##pdb.set_trace()
    if len(references) > 0:
        messages = inject_references_to_messages(messages, references)

    return await async_generate_fn(client, model, messages, temperature, max_tokens)

@register_model("moa_together")
class MoA_Together(APIModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        self.async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))
        self.reference_models = [
            "Qwen/Qwen1.5-110B-Chat",
            "Qwen/Qwen1.5-72B-Chat",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
            "databricks/dbrx-instruct",
			"microsoft/WizardLM-2-8x22B",
			"meta-llama/Llama-3-70b-chat-hf"
        ]

        self.aggregator_model = "Qwen/Qwen1.5-110B-Chat"
        
        self.num_rounds = 2
        self.MAX_NEW_TOKENS = 1024
        self.FIX_INTERVAL_SECOND = 1
        self.MAX_RETRY_NUM = 10

    async def generate_moa_response(self, prompt):
        ##pdb.set_trace()
        messages = [{"role": "user", "content": prompt}]
        references = []

        if self.reference_models:
            prev_references = []

            for i_round in range(self.num_rounds):
                references = []

                for reference_model in self.reference_models:
                    reference = await generate_with_references(
                        client=self.async_client,
                        model=reference_model,
                        messages=messages,
                        references=prev_references,
                        temperature=0.7,
                        max_tokens=self.MAX_NEW_TOKENS,
                    )

                    if reference is not None:
                        references.append(reference)

                ##pdb.set_trace()

                if i_round < self.num_rounds - 1:
                    prev_references = references
                    references = []

        output = await generate_with_references(
            client=self.async_client,
            model=self.aggregator_model,
            messages=messages,
            references=references,
            temperature=0.7,
            max_tokens=self.MAX_NEW_TOKENS,
        )

        output = output.strip()

        messages.append(
            {
                "role": "assistant",
                "content": output,
            }
        )

        return output

    def _decode(self, inputs):
        prompt = inputs[0]["content"]
        response = asyncio.run(self.generate_moa_response(prompt))
        time.sleep(self.FIX_INTERVAL_SECOND)
        return response

    def decode(self, inputs):
        for i in range(self.MAX_RETRY_NUM):
            try:
                response_content = self._decode(inputs)
                return response_content
            except Exception as e:
                print("Error in decode, retrying...")
                print(e)
                time.sleep(1)
                continue
        print("Failed after retries.")
        return 'Error'
