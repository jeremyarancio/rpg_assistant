import argparse
import logging
import os
import shutil
import sys
from distutils.util import strtobool

import torch.cuda
from datasets import load_from_disk
from peft import PeftModel  # for typing only
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, PreTrainedModel,
                          Trainer, TrainingArguments, set_seed)


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def copy_inference_script(path: str) -> None:
    """Copy inference script to model directory for later deployment. `inference.py` is loaded into the instance along
    `train.py` and `requirements.txt` during training.

    Args:
        path (str): SM_MODEL_DIR / code
    """
    os.makedirs(path, exist_ok=True)
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "inference.py"),
        os.path.join(path, "inference.py"),
    )
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "requirements.txt"),
        os.path.join(path, "requirements.txt"),
    )


def parse_args():
    parser = argparse.ArgumentParser()

    # Sagemaker environment
    parser.add_argument("--dataset_dir", type=str, default=os.getenv("SM_CHANNEL_TRAINING"))
    parser.add_argument("--output_dir", type=str, default=os.getenv("SM_MODEL_DIR"))

    #Training
    parser.add_argument("--pretrained_model_name", type=str, default="bigscience/bloom-3b", help="Name of the pretrained model to fine-tune.")
    parser.add_argument("--epochs", type=float, default=1, help="Number of epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--gradient_checkpointing", type=strtobool, default=True, help="Gradient checkpointing")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps to save memory.")
    parser.add_argument("--bf16", type=strtobool, default=True if torch.cuda.get_device_capability()[0] == 8 else False, help="Whether to use bf16.")
    parser.add_argument("--merge_weights", type=strtobool, default=False, help="Whether to merge LoRA weights with base model.")
    
    #Lora
    parser.add_argument('--r', type=int, default=32, help='Number of attention heads for LoRA.')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha parameter.')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout parameter.')
    
    args = parser.parse_known_args()
    return args


def train(args):

    set_seed(args.seed)
    
    tokenizer = load_tokenizer(args.pretrained_model_name)
    use_cache = False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True,
        use_cache=use_cache
    )
    LOGGER.info(f"Pretrained model imported from {args.pretrained_model_name} and tokenizer imported from {args.pretrained_model_name}")
    LOGGER.info(f'Prepare model: freeze pretrained model - gradient_checkpointng = {args.gradient_checkpointing} - cast layer norms and head to Float32')
    model = prepare_model(model, gradient_checkpointing=args.gradient_checkpointing)
    LOGGER.info(f"Create LoRA model.")
    model = create_peft_model(model=model, r=args.r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

    LOGGER.info(f"Load dataset from {args.dataset_dir}.")
    dataset = load_from_disk(args.dataset_dir)
    LOGGER.info(f"Number of tokens for the training: {dataset.num_rows*len(dataset['input_ids'][0])}")

    LOGGER.info("Start training.")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        # logging strategies
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        # checkpoints
        save_strategy="steps",
        save_total_limit=1, # only save the last checkpoint and delete the previous ones
        save_steps=0.2 # Portion of the total training steps
    )
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    trainer.train()
    LOGGER.info(f"Training done. Start saving with merge_weights = {args.merge_weights}.")

    if args.merge_weights:
        LOGGER.info(f"Merge LoRA weights with base model.")
        # merge adapter weights with base model and save
        # save int 4 adapters
        trainer.model.save_pretrained(args.output_dir, safe_serialization=False)
        # clear memory
        del model
        del trainer
        torch.cuda.empty_cache()

        from peft import AutoPeftModelForCausalLM

        # load PEFT model in fp16
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.output_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True  # ATTENTION: This allows remote code execution
        )  
        # Merge LoRA and base model and save
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    else:
        LOGGER.info(f"Save model without merging weights.")
        trainer.model.save_pretrained(args.output_dir, safe_serialization=False)
    tokenizer.save_pretrained(args.output_dir)


def prepare_model(model: PreTrainedModel, gradient_checkpointing: bool) -> PreTrainedModel:
    """Prepare model for PEFT - QLoRA"""
    from torch import bfloat16, float16, float32

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


def create_peft_model(model: PreTrainedModel, r: int, lora_alpha: int, lora_dropout: float) -> PeftModel:
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    )
    # prepare int-8 model for training
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_tokenizer(pretrained_model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Bloom is left padded / Falcon right padded
    return tokenizer


if __name__ == "__main__":

    args, _ = parse_args()
    train(args)
    copy_inference_script(os.path.join(args.output_dir, "code"))
