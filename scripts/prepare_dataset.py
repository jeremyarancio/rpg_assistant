from typing import Mapping
import logging

from datasets import load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizer

from config.config import ConfigFireball, ConfigTraining


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def prepare_dataset(dataset_path: str, pretrained_model_name: str) -> None:
   dataset = load_from_disk(ConfigFireball.save_to_disk_dir / "fireball_postprocessed")
   dataset.cleanup_cache_files()
   dataset = dataset.map(group_prompt_prediction, remove_columns=["prompt", "prediction"])
   tokenizer = load_tokenizer(pretrained_model_name)
   dataset = dataset.map(tokenize, remove_columns="text", fn_kwargs={"tokenizer": tokenizer})
   max_length = len(max(dataset["input_ids"], key=len))
   LOGGER.info(f"Maximal length in the dataset: {max_length}") # Max_length = 2788
   # We save the tokenizer with the new tokens
   dataset.push_to_hub("JeremyArancio/fireball_tokenized", private=True)


def group_prompt_prediction(element: Mapping) -> Mapping:
   """ Group prompt and prediction in a single separated by a specifc token"""
   element["text"] = element["prompt"] + ConfigFireball.PREDICTION_KEY + element["prediction"]
   return element


def load_tokenizer(pretrained_model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize(element: Mapping, tokenizer: PreTrainedTokenizer) -> Mapping:
   inputs = tokenizer(
      element["text"], 
      truncation=True, 
      return_length=True, 
      max_length=ConfigTraining.max_length
   )
   return inputs


if __name__ == "__main__":

   prepare_dataset(
      dataset_path=ConfigFireball.dataset_hf_repo,
      pretrained_model_name=ConfigTraining.pretrained_model_name
   )