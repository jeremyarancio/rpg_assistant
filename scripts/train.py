import logging
from typing import Mapping
import argparse

from datasets import load_dataset, Dataset
from peft import PeftModel # for typing only
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorForLanguageModeling, 
    TrainingArguments, 
    Trainer,
    PreTrainedModel
)



LOGGER = logging.getLogger(__name__)


def train(
        pretrained_model_name: str, 
        gradient_checkpointing: bool,
        lora_config: Mapping,
        trainer_config: Mapping,
        mlm: bool,
    ) -> None:

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    use_cache = False if gradient_checkpointing else True,  # this is needed for gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name, 
        device_map="auto", 
        load_in_4bit=True,
        trust_remote_code=True,
        use_cache=use_cache
    )
    LOGGER.info(f"Pretrained model and tokenizer imported from {pretrained_model_name}")
    tokenizer.pad_token = tokenizer.eos_token
    model = prepare_model(model)
    model = create_peft_model(model, lora_config)
    dataset = load_dataset
    dataset = load_dataset(hf_repo, split)
    LOGGER.info(f"Train dataset downloaded:\n {dataset['train']}")
    LOGGER.info(f"Number of tokens for the training: {dataset['train'].num_rows*len(dataset['train']['input_ids'][0])}")
    trainer = Trainer(
        model=model,
        train_dataset=dataset['train'],
        args=TrainingArguments(**trainer_config),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=mlm)
    )
    trainer.train()
    model.push_to_hub(repo_id=hf_repo)
    tokenizer.push_to_hub(repo_id=hf_repo)


def prepare_model(model: PreTrainedModel, gradient_checkpointing: bool) -> PreTrainedModel:
    from torch import float16, bfloat16, float32

    # freeze the model
    for param in model.parameters():
      param.requires_grad = False
      # cast all non INT8 parameters to fp32
      if (param.dtype == float16) or (param.dtype == bfloat16):
        param.data = param.data.to(float32)
    # reduce number of stored activations
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  
    model.enable_input_require_grads()
    return model


def create_peft_model(model: PreTrainedModel, lora_config: Mapping) -> PeftModel:
    from peft import get_peft_model, LoraConfig, TaskType

    peft_config = LoraConfig(**lora_config)
        # task_type=TaskType.CAUSAL_LM,
        # inference_mode=False,
        # r=8,
        # lora_alpha=32,
        # lora_dropout=0.05,
        # target_modules=["query_key_value"]

    # prepare int-8 model for training
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

