import os
import asyncio
import time
import copy
from together import AsyncTogether, Together
from mix_eval.models.base_api import APIModelBase
from mix_eval.api.registry import register_model
import pdb
import sys
from pathlib import Path
import json

print("Starting to execute the file")
# Your imports here
print("Imports completed")
# Rest of your code
print("About to define Archon_Together class")

# Add the parent directory of `archon` to sys.path
parent_dir = str(Path(__file__).resolve().parents[4])  # Adjust the number as per your directory depth
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from archon import Archon
json_config_path = os.path.join(parent_dir, "configs", "archon-70Bx8_1_samples_then_critic_then_70Bx8_layer_then_fuser_with_Qwen2_72B.json")

# Define the relative path to the JSON configuration file based on the `archon` directory

print("LOG: Loading Archon configuration from", json_config_path)
with open(json_config_path, 'r') as f:
    config = json.load(f)


@register_model("archon_together")
class Archon_Together(APIModelBase):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        archon_config = config
        self.archon = Archon(config=archon_config)
        
    def decode(self, inputs):
        response = self.archon.generate(inputs)
        return response

print("Archon_Together model registered")