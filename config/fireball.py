import os
from pathlib import Path


REPO_DIR = Path(os.path.dirname(os.path.realpath(__file__))).parent


class ConfigFireball():
    
    dataset_hf_repo = "JeremyArancio/fireball"
    save_to_disk_dir = REPO_DIR / "data/fireball/"

    prompt_template = """
    Last utterance:
    {before_utterances}

    Command description:
    {command_description}

    History:
    {utterance_history}
    """