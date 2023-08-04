import os
import logging
from pathlib import Path

from peft import TaskType


REPO_DIR = Path(os.path.dirname(os.path.realpath(__file__))).parent
LOGGER = logging.getLogger(__name__)


class ConfigFireball():
    
    fireball_dataset = "JeremyArancio/fireball"
    data_dir = REPO_DIR / "data"
    s3_data_uri = "s3://rpg-assistant/fireball_data/fireball_tokenized"
    PREDICTION_KEY = "\n### Prediction:\n"

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

    pretrained_model_name = "bigscience/bloom-3b"
    dataset_path = "JeremyArancio/fireball_tokenized" #TODO: remove
    model_name = "JeremyArancio/rpg-assistant-v1" #TODO: remove
    output_dir = "./tmp/model" #TODO: remove
    model_save_dir = "/opt/ml/model/" #TODO: remove
    max_length = 500 #TODO: move to ConfigFireball
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

  