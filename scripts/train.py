import logging
from typing import Mapping

from datasets import load_dataset, Dataset
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
        lora_config: Mapping,
        trainer_config: Mapping,
        mlm: bool,
    ) -> None:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name, 
        device_map="auto", 
        load_in_4bit=True,
        trust_remote_code=True
    )
    model = prepare_model(model)
    model = get_peft_model(model, LoraConfig(**lora_config))
    LOGGER.info(f"Model trainable parameters:\n {print_trainable_parameters(model)}")
    dataset = load_dataset(hf_repo)
    LOGGER.info(f"Train dataset downloaded:\n {dataset['train']}")
    LOGGER.info(f"Number of tokens for the training: {dataset['train'].num_rows*len(dataset['train']['input_ids'][0])}")
    trainer = Trainer(
        model=model,
        train_dataset=dataset['train'],
        args=TrainingArguments(**trainer_config),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=mlm)
    )
    model.config.use_cache = False  # silence warnings
    trainer.train()
    model.config.use_cache = True
    model.push_to_hub(repo_id=hf_repo)
    tokenizer.push_to_hub(repo_id=hf_repo)


def prepare_model(model: PreTrainedModel, gradient_checkpointing: bool, 
                  lora_config: Mapping) -> PreTrainedModel:
    """Prepare model with QLora
    """
    from peft import get_peft_model, LoraConfig, prep

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def create_peft_config()
    
