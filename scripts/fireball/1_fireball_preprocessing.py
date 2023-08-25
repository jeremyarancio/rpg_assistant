import logging
from typing import Mapping
import re

from datasets import Dataset, load_from_disk, disable_caching

from scripts.config import ConfigFireball


LOGGER = logging.getLogger(__name__)
disable_caching()


def preparation(dataset: Dataset) -> Dataset:
    dataset.cleanup_cache_files()
    data = dataset.filter(filter_function)
    data = data.map(mapping_function, remove_columns=dataset.column_names)
    return data


def mapping_function(element: Mapping) -> Mapping:
    element = remove_player(element)
    element = preprocess_text(element)
    element = fill_missing_before_utterances(element)
    element = remove_command_from_utterance_history(element)
    element = transform_to_prompt_prediction(element)
    return element


def fill_missing_before_utterances(element: Mapping) -> Mapping:
    if element['before_utterances'] == []:
        element['before_utterances'] = [element["utterance_history"][-1]]
    return element


def preprocess_text(element: Mapping) -> Mapping:
    """Remove * from utterances
    """
    element["utterance_history"] = [utterance.replace("*", "") for utterance in element["utterance_history"]]
    element["before_utterances"] = [utterance.replace("*", "") for utterance in element["before_utterances"]]
    element["after_utterances"] = [utterance.replace("*", "") for utterance in element["after_utterances"]]
    return element


def remove_player(element: Mapping) -> Mapping:
    """Remove player string from utterances history
    Examples:
    ```markdown
    "Player 3 of Twilight [6]: Player 3's darts shoot from his hand, 3 hit the chief in a row, impaling his skull and spilling his brains onto the grass...The 4th dart hits The orc in front of him squarly in the chest and leaves a singe mark on his armor.",
    'Fredbear (Zal 6)(Player 2 6): ‘She begin to put both her javelin and her shield away into her back while inrage then brought a big great axe and go for a double strike against this or4’',
    ```
    """
    element["utterance_history"] = [re.sub("^.+:\s*", "", utterance) for utterance in element["utterance_history"]]
    return element


def remove_command_from_utterance_history(element: Mapping) -> Mapping:
    element["utterance_history"] = [utterance for utterance in element["utterance_history"] if not re.match("^!", utterance)]
    return element


def filter_function(element: Mapping) -> bool:
    """
    Filter all events that:
    * Have no after_utterances OR
    * Have no before_utterances AND no utterance_history
    """
    return element["after_utterances"] != [] and \
        any([element['before_utterances'] != [], element['utterance_history'] != []])


def transform_to_prompt_prediction(element: Mapping) -> Mapping:
    before_utterances = "\n\n".join(element["before_utterances"])
    command_description = "\n\n".join(element["automation_results"])
    utterance_history = "\n\n".join(element["utterance_history"])
    after_utterances = "\n\n".join(element["after_utterances"])
    # Prepare the prompt
    prompt = ConfigFireball.prompt_template.format(
        before_utterances=before_utterances,
        command_description=command_description,
        utterance_history=utterance_history,
    )
    element["prompt"] = prompt
    element['prediction'] = after_utterances
    return element


def main():
    dataset = load_from_disk(dataset_path=ConfigFireball.fireball_path)
    dataset_v1 = preparation(dataset)
    dataset_v1.save_to_disk(dataset_path=ConfigFireball.fireball_postprocessed_path)


if __name__ == "__main__":

    main()
