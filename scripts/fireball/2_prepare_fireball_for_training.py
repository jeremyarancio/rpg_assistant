from typing import Mapping
import logging

from datasets import load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizer

from scripts.config import ConfigFireball, ConfigTraining


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def prepare_dataset() -> None:
   dataset = load_from_disk(ConfigFireball.fireball_postprocessed_path)
   dataset.cleanup_cache_files()
   dataset = dataset.map(group_prompt_prediction, remove_columns=["prompt", "prediction"])
   tokenizer = load_tokenizer(ConfigTraining.pretrained_model_name)
   dataset = dataset.map(tokenize, remove_columns="text", fn_kwargs={"tokenizer": tokenizer})
   # We remove truncated sequences (See notebooks/fireball_dataset/3_training_optimization.ipynb)
   dataset = dataset.filter(lambda x: x["input_ids"][-1] == tokenizer.pad_token_id) 
   dataset.save_to_disk(ConfigFireball.fireball_tokenized_path)


def group_prompt_prediction(element: Mapping) -> Mapping:
   """ Group prompt and prediction in a single separated by a specifc token"""
   element["text"] = element["prompt"] + ConfigFireball.PREDICTION_KEY + element["prediction"]
   return element


def load_tokenizer(pretrained_model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Bloom is left padded / Falcon right padded
    return tokenizer


def tokenize(element: Mapping, tokenizer: PreTrainedTokenizer) -> Mapping:
   inputs = tokenizer(
      element["text"], 
      truncation=True,
      max_length=ConfigFireball.max_length,
      padding="max_length",
   )
   return inputs


if __name__ == "__main__":

   prepare_dataset()