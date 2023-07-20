import logging
from typing import Mapping
import argparse

from datasets import load_dataset, Dataset
from peft import PeftModel # for typing only
import torch.cuda
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    set_seed, 
    DataCollatorForLanguageModeling, 
    TrainingArguments, 
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizer
)

from config.config import ConfigTraining


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()

    # add model id and dataset path argument
    parser.add_argument("--model_id", type=str, default=ConfigTraining.pretrained_model_name, 
                        help="Model id to use for training.")
    parser.add_argument("--dataset_path", type=str, default=ConfigTraining.lm_dataset, help="Path to dataset.")
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=ConfigTraining.epochs, help="Number of epochs to train for.")
    parser.add_argument("--per_device_train_batch_size", type=int, 
                        default=ConfigTraining.per_device_batch_size, help="Batch size to use for training.")
    parser.add_argument("--lr", type=float, default=ConfigTraining.lr, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=ConfigTraining.seed, help="Seed to use for training.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=ConfigTraining.gradient_checkpointing, 
                        help="Path to deepspeed config file.")
    parser.add_argument("--bf16", type=bool, 
                        default=True if torch.cuda.get_device_capability()[0] == 8 else False, help="Whether to use bf16.")
    parser.add_argument("--merge_weights", type=bool, default=ConfigTraining.merge_weights, 
                        help="Whether to merge LoRA weights with base model.")
    
    args = parser.parse_known_args()
    return args


def train(args):

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    use_cache = False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        device_map="auto", 
        load_in_4bit=True,
        trust_remote_code=True,
        use_cache=use_cache
    )
    LOGGER.info(f"Pretrained model and tokenizer imported from {args.model_id}")
    model = prepare_model(model, gradient_checkpointing=args.gradient_checkpointing)
    model = create_peft_model(model)
    
    dataset = load_dataset(args.lm_dataset, split="train")
    LOGGER.info(f"Number of tokens for the training: {dataset.num_rows*len(dataset['input_ids'][0])}")
    
    training_args = TrainingArguments(
        output_dir=ConfigTraining.output_dir / "tmp",
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        # logging strategies
        logging_dir=f"{ConfigTraining.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
    )
    trainer = Trainer(
        model=model,
        train_dataset=dataset['train'],
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    trainer.train()
    save_model(args.merge_weights, trainer, tokenizer)

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


def create_peft_model(model) -> PeftModel:

    from peft import get_peft_model, LoraConfig, TaskType

    lora_config = LoraConfig(
        task_type=ConfigTraining.task_type,
        inference_mode=ConfigTraining.inference_mode,
        r=ConfigTraining.r,
        lora_alpha=ConfigTraining.lora_alpha,
        lora_dropout=ConfigTraining.lora_dropout,
        target_modules=ConfigTraining.target_modules
    )
    # prepare int-8 model for training
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def save_model(merge_weights: bool, trainer: Trainer, tokenizer: PreTrainedTokenizer) -> None:
    """Save model with our without merging Adapters from Lora"""

    if merge_weights:
        # merge adapter weights with base model and save
        # save int 4 model
        trainer.model.save_pretrained(ConfigTraining.output_dir / "tmp", safe_serialization=False)
        # clear memory
        del model
        del trainer
        torch.cuda.empty_cache()

        from peft import AutoPeftModelForCausalLM

        # load PEFT model in fp16
        model = AutoPeftModelForCausalLM.from_pretrained(
            ConfigTraining.output_dir / "tmp",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True  # ATTENTION: This allows remote code execution
        )  
        # Merge LoRA and base model and save
        merged_model = model.merge_and_unload()
        # merged_model.save_pretrained(ConfigTraining.output_dir, safe_serialization=True) #TODO: uncomment when working with Sagemaker
        merged_model.push_to_hub(repo_id=ConfigTraining.model_name)
    else:
        # trainer.model.save_pretrained(ConfigTraining.output_dir, safe_serialization=True) #TODO: uncomment when working with Sagemaker
        trainer.push_to_hub(repo_id=ConfigTraining.model_name)

    tokenizer.push_to_hub(repo_id=ConfigTraining.model_name)



