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
    pretrained_model_name = "tiiuae/falcon-7b"
    dataset_path = "JeremyArancio/fireball_tokenized"
    model_name = "JeremyArancio/mpt-7b-QLora-4bits-rpg-assistant-v1"
    output_dir = REPO_DIR / "models"
    max_length = 3000 # Max length in Fireball: ~2800
    epochs = 0.1
    per_device_batch_size = 4
    lr = 5e-5
    seed = 42
    merge_weights = True
    gradient_checkpointing = True
    gradient_accumulation_steps = 4

    #peft - lora
    task_type = TaskType.CAUSAL_LM
    inference_mode = False
    r = 64
    lora_alpha = 16
    lora_dropout = 0.05
  