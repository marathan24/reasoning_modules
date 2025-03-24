import logging
import os
from dotenv import load_dotenv
from typing import Dict, List
import asyncio
import json, random

from naptha_sdk.inference import InferenceClient
from naptha_sdk.schemas import AgentDeployment, AgentRunInput, KBRunInput
from naptha_sdk.user import sign_consumer_id, get_private_key_from_pem

from reasoning_modules.schemas import ReasoningInput, SystemPromptSchema
from reasoning_modules.prompt import cot_prompt, standard_prompt

logger = logging.getLogger(__name__)

class ReasoningAgent:
    async def create(self, deployment: AgentDeployment, *args, **kwargs):
        self.deployment = deployment
        self.system_prompt = SystemPromptSchema(role=self.deployment.config.system_prompt["role"])
        self.inference_client = InferenceClient(self.deployment.node)
    
    async def run(self, module_run: AgentRunInput, *args, **kwargs):
        problem = module_run.inputs.problem
        num_thoughts = module_run.inputs.num_thoughts if hasattr(module_run.inputs, 'num_thoughts') else 3
        
        # Generate multiple reasoning paths using ToT approach
        thoughts = []
        for i in range(num_thoughts):
            # Use CoT prompt directly instead of initial generation as done in last commit
            cot_prompt_text = cot_prompt.format(input=problem)
            messages = [
                {"role": "system", "content": self.system_prompt.role},
                {"role": "user", "content": cot_prompt_text}
            ]
            logger.info(f"Generating thought {i+1}/{num_thoughts} with CoT prompt: %s", messages)
            
            response_cot = await self.inference_client.run_inference({
                "model": self.deployment.config.llm_config.model,
                "messages": messages,
                "temperature": self.deployment.config.llm_config.temperature,
                "max_tokens": self.deployment.config.llm_config.max_tokens
            })
            
            cot_thought = response_cot.choices[0].message.content
            logger.info(f"Generated thought {i+1}: %s", cot_thought)
            thoughts.append(cot_thought)
        
        return {
            "thoughts": thoughts,
            "problem": problem
        }

async def run(module_run: Dict, *args, **kwargs):
    module_run = AgentRunInput(**module_run)
    module_run.inputs = ReasoningInput(**module_run.inputs)
    reasoning_agent = ReasoningAgent()
    await reasoning_agent.create(module_run.deployment)
    result = await reasoning_agent.run(module_run)
    return result

if __name__ == "__main__":
    import asyncio
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment

    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    naptha = Naptha()

    deployment = asyncio.run(
        setup_module_deployment(
            "agent", 
            "reasoning_modules/configs/deployment.json", 
            node_url=os.getenv("NODE_URL"), 
            user_id=naptha.user.id
        )
    )

    # with open('./reasoning_modules/test.jsonl', 'r') as f:
    #     lines = f.readlines()

    # if lines:
    #     data = json.loads(random.choice(lines))
    #     question_text = data.get("question", "Prove that the sum of the angles in a triangle is 180 degrees.")
    # else:
    #     question_text = "Prove that the sum of the angles in a triangle is 180 degrees."

    input_params = {
        "func_name": "reason",
        "problem": "question_text",
        "num_thoughts": 3  # Default number of thoughts to generate
    }

    module_run = {
        "inputs": input_params,
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, get_private_key_from_pem(os.getenv("PRIVATE_KEY")))
    }

    response = asyncio.run(run(module_run))

    logger.info("Final Reasoning Output:")
    logger.info(response)