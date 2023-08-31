import os
import logging
from pathlib import Path

from peft import TaskType


REPO_DIR = Path(os.path.dirname(os.path.realpath(__file__))).parent
LOGGER = logging.getLogger(__name__)


class ConfigFireball():
    
    fireball_path = REPO_DIR / "data/fireball"
    fireball_postprocessed_path = REPO_DIR / "data/fireball_postprocessed"
    fireball_tokenized_path = REPO_DIR / "data/fireball_tokenized"
    s3_data_uri = f"s3://rpg-assistant/fireball_data/fireball_tokenized"
    PREDICTION_KEY = "\n### Prediction:\n"
    max_length = 500
    prompt_template =  (
        "### Last utterance:\n" 
        + "{before_utterances}" 
        + "\n### Command description:\n" 
        + "{command_description}" 
        + "\n### History:\n" 
        + "{utterance_history}"
    )


class ConfigTraining():
    job_name = "bloom3B-qlora-fireball"
    bucket_name = "rpg-assistant"
    pretrained_model_name = "bigscience/bloom-3b"
    instance_type = 'ml.g4dn.xlarge'
    instance_count = 1
    epochs = 1
    per_device_batch_size = 4
    lr = 5e-5
    seed = 42
    merge_weights = True
    gradient_checkpointing = True
    gradient_accumulation_steps = 4

    #peft - lora
    task_type = TaskType.CAUSAL_LM
    inference_mode = False
    r = 32
    lora_alpha = 16
    lora_dropout = 0.05

  