import os
from pathlib import Path


REPO_DIR = Path(os.path.dirname(os.path.realpath(__file__))).parent


class ConfigFireball():
    
    dataset_hf_repo = "JeremyArancio/fireball"
    save_to_disk_dir = REPO_DIR / "data/fireball/"
    LAST_UTTERANCE_KEY = "### Last utterance:\n"
    COMMAND_DESCRIPTION_KEY = "### Command description:\n"
    UTTERANCE_HISTORY_KEY = "### History:\n"
    PREDICTION_KEY = "### Prediction:\n"

    prompt_template = """\
{LAST_UTTERANCE_KEY}\
{before_utterances}\
{COMMAND_DESCRIPTION_KEY}\
{command_description}\
{UTTERANCE_HISTORY_KEY}\
{utterance_history}\
"""


class ConfigTraining():
    pretrained_model_name = "mosaicml/mpt-7b"
    model_name = "JeremyArancio/mpt-7b-QLora-4bits-rpg-assistant-v1"
    max_length = 3400