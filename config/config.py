import os
from pathlib import Path
from peft import TaskType


REPO_DIR = Path(os.path.dirname(os.path.realpath(__file__))).parent


class ConfigFireball():
    
    dataset_hf_repo = "JeremyArancio/fireball"
    save_to_disk_dir = REPO_DIR / "data"
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
    pretrained_model_name = "mosaicml/mpt-7b"
    lm_dataset = "JeremyArancio/fireball_tokenized"
    model_name = "JeremyArancio/mpt-7b-QLora-4bits-rpg-assistant-v1"
    output_dir = REPO_DIR / "models"
    max_length = 3400 # Max length in Fireball: ~2800
    epochs = 3
    per_device_batch_size = 1
    lr = 5e-5
    seed = 42
    merge_weights = True

    #peft - lora
    task_type = TaskType.CAUSAL_LM,
    inference_mode = False,
    r = 8,
    lora_alpha = 32,
    lora_dropout = 0.05,
    target_modules = ["query_key_value"]