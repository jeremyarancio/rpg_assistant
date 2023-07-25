import os
import logging
from pathlib import Path

from peft import TaskType
import sagemaker


REPO_DIR = Path(os.path.dirname(os.path.realpath(__file__))).parent
LOGGER = logging.getLogger(__name__)


# Init Sagemaker
sess = sagemaker.Session()
sagemaker_session_bucket = sess.default_bucket()
sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)
LOGGER.info(f"sagemaker bucket: {sess.default_bucket()}")
LOGGER.info(f"sagemaker session region: {sess.boto_region_name}")


class ConfigFireball():
    
    fireball_dataset = "JeremyArancio/fireball"
    data_dir = REPO_DIR / "data"
    s3_bucket_uri = f"s3://{sess.default_bucket()}/rpg-assistant/fireball_data"
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
    pretrained_model_name = "bigscience/bloom-3b"
    dataset_path = "JeremyArancio/fireball_tokenized"
    model_name = "JeremyArancio/rpg-assistant-v1"
    output_dir = "./tmp/model"
    model_save_dir = "/opt/ml/model/"
    max_length = 500
    epochs = 1
    per_device_batch_size = 4
    lr = 5e-5
    seed = 42
    merge_weights = False
    gradient_checkpointing = True
    gradient_accumulation_steps = 4

    #peft - lora
    task_type = TaskType.CAUSAL_LM
    inference_mode = False
    r = 64
    lora_alpha = 16
    lora_dropout = 0.05
    target_modules=["query_key_value"]
  