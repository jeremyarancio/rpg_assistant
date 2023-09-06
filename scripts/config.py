import os
import logging
from pathlib import Path

from peft import TaskType


REPO_DIR = Path(os.path.dirname(os.path.realpath(__file__))).parent
PROJECT_NAME = "rpg-assistant"


class ConfigFireball:
    
    fireball_path = REPO_DIR / "data/fireball"
    fireball_postprocessed_path = REPO_DIR / "data/fireball_postprocessed"
    fireball_tokenized_path = REPO_DIR / "data/fireball_tokenized"
    s3_data_uri = f"s3://{PROJECT_NAME}/fireball_data/fireball_tokenized"
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


class ConfigTraining:
    job_name = "bloom3B-qlora-fireball"
    bucket_name = PROJECT_NAME
    pretrained_model_name = "bigscience/bloom-3b"
    instance_type = 'ml.g4dn.xlarge'
    instance_count = 1
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
    r = 32
    lora_alpha = 16
    lora_dropout = 0.05

    #Sagemaker estimator
    transformers_version = "4.28"
    pytorch_version = "2.0"
    py_version = "py310"
    source_dir_folder_name = "sagemaker_scripts/"
    entry_point = "train.py"


class ConfigModelRegistry:
    prefix_name = "models-registry"
    training_job_name = "bloom3B-qlora-fireball-2023-08-05-15-57-44-467"
    artifact_uri = f"s3://{PROJECT_NAME}/{prefix_name}/{training_job_name}/output/model.tar.gz "


class ConfigRegistry:
    model_data_uri =  f"s3://rpg-assistant/training-jobs/bloom3B-qlora-fireball-2023-09-05-17-58-13-796/output/model.tar.gz"
    inference_instance_type = 'ml.g4dn.xlarge'
    batch_instance_type = 'ml.g4dn.xlarge'
    instance_count = 1
    endpoint_name = "test-inference-endpoint"
    model_package_group_name = "fireball-llms"
    model_name = "bloom3b-qlora-fireball"
    approval_status = "PendingManualApproval"
    description = "Next utterance generation"
    test_data = {
        "inputs": """### Last utterance:
Razored teeth lash out to take advantage of an opening
### Command description:
Lisbeth Ethuil uses Polearm Master - Bonus Attack!

### History:
The creature, a living crucible of coldest winter, rears back; there is a hitching in its neck, and then it vomits out a spray of biting ice...

There is a low hiss; it seems to realise its ice will be no use. Claws like scythes sink into the frozen earth below and flex in readiness...

Said only movement of Lis' turn, will assume applies)

moving forward, only allowing range for herself, her halberd swings out with practiced, deadly precision.

Razored teeth lash out to take advantage of an opening

### Prediction:
"""
    }

  