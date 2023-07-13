

class ConfigFireball():
    
    dataset_hf_repo = "JeremyArancio/fireball"

    prompt_template = """Last utterance:
    {before_utterances}

    Command description:
    {command_description}

    History:
    {utterance_history}
    """