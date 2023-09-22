import logging
from typing import Mapping, Optional
import json

import boto3
from transformers import AutoTokenizer

from scripts.config import ConfigDeployment, ConfigTraining


LOGGER = logging.getLogger(__name__)


class FireballInference:

    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained(ConfigTraining.pretrained_model_name)
        self.eos_token_ids = tokenizer.eos_token_id
        self.endpoint_name = ConfigDeployment.endpoint_name

    def generate_next_utterance(
            self,
            last_utterance: str,
            command_descr: str,
            history: str,
            hyperparameters: Mapping
        ) -> str:
        """_summary_

        Args:
            last_utterance (str): _description_
            command_descr (str): _description_
            history (str): _description_
            hyperparameters (Mapping): _description_

        Returns:
            str: _description_
        """
        prompt = PromptBuilder.build_prompt_for_next_utterance_prediction(
            last_utterance,
            command_descr,
            history
        )
        LOGGER.info(f"Prompt: {prompt}")
        # The eos_token_id corresponding to the llm is added to the hyperparameters 
        hyperparameters["eos_token_id"] = self.eos_token_ids
        payload = json.dumps({"inputs": prompt, "parameters": hyperparameters})
        LOGGER.info(f"Start invoking the endpoint")
        runtime = boto3.client('runtime.sagemaker')
        response = runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=payload,
        )
        result = json.loads(response['Body'].read().decode())
        LOGGER.info(f"Decoded result from response: {result}")
        next_utterance = result['generated_text']
        # Remove prompt from the generated text
        post_processed_next_utterance = next_utterance.replace(prompt, "")
        LOGGER.info(f"Next utterance generated: {post_processed_next_utterance}")
        return post_processed_next_utterance
    

class PromptBuilder:

    def build_prompt_for_next_utterance_prediction(
            last_utterance: str,
            command_descr: str,
            history: str
    ) -> str:
        prompt = (
            "### Last utterance:\n" +
            last_utterance +
            "\n### Command description:\n" +
            command_descr +
            "\n### History:\n" +
            history +
            "\n### Prediction:\n"
        )
        return prompt


if __name__ == "__main__":
    # Testing
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    last_utterance = "Razored teeth lash out to take advantage of an opening"
    command_descr = "Lisbeth Ethuil uses Polearm Master - Bonus Attack!"
    history = """The creature, a living crucible of coldest winter, rears back; there is a hitching in its neck, and then it vomits out a spray of biting ice...

There is a low hiss; it seems to realise its ice will be no use. Claws like scythes sink into the frozen earth below and flex in readiness...

Said only movement of Lis' turn, will assume applies)

moving forward, only allowing range for herself, her halberd swings out with practiced, deadly precision.
"""
    hyperparameters = {
        "max_new_tokens": 50,
        # "do_sample": False,
        "temperature": 0.1,
        "early_stopping": True,
        "repetition_penalty": float(5),
        "no_repeat_ngram_size": 3,
        "num_beams": 3,
    }

    predictor = FireballInference()
    print(predictor.generate_next_utterance(
        last_utterance,
        command_descr,
        history,
        hyperparameters
    ))
