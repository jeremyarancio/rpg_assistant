

from datasets import load_dataset


def prepare_dataset(dataset_path: str):
   
   dataset = load_dataset(dataset_path, split="train")
   
