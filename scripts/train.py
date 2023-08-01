import os
import logging
import argparse

from datasets import load_from_disk
from peft import PeftModel # for typing only
import torch.cuda
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    set_seed, 
    DataCollatorForLanguageModeling, 
    TrainingArguments, 
    Trainer,
    PreTrainedModel
)

from config import ConfigTraining


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # set your logging level
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
    filename=os.environ["SM_MODEL_DIR"] + '/log_filename.log',
    filemode='w'
)


def parse_args():
    parser = argparse.ArgumentParser()

    # Need to be arg
    parser.add_argument("--dataset_dir", type=str, help="Path to the training dataset.")

    parser.add_argument("--output_dir", type=str, default=ConfigTraining.output_dir)
    parser.add_argument("--pretrained_model_name", type=str, default=ConfigTraining.pretrained_model_name)
    parser.add_argument("--model_name", type=str, default=ConfigTraining.model_name, help="Pretrained model from the hub to use for training.")
    parser.add_argument("--epochs", type=float, default=ConfigTraining.epochs, help="Number of epochs to train for.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=ConfigTraining.per_device_batch_size, help="Batch size to use for training.")
    parser.add_argument("--lr", type=float, default=ConfigTraining.lr, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=ConfigTraining.seed, help="Seed to use for training.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=ConfigTraining.gradient_checkpointing, help="Gradient checkpointing")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=ConfigTraining.gradient_accumulation_steps, help="Number of gradient accumulation steps to save memory.")
    parser.add_argument("--bf16", type=bool, default=True if torch.cuda.get_device_capability()[0] == 8 else False, help="Whether to use bf16.")
    parser.add_argument("--merge_weights", type=bool, default=ConfigTraining.merge_weights, help="Whether to merge LoRA weights with base model.")

    args = parser.parse_known_args()
    return args


def train(args):

    set_seed(args.seed)

    # The tokenizer was prepared during the dataset preparation
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    use_cache = False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True,
        use_cache=use_cache
    )
    LOGGER.info(f"Pretrained model imported from {args.pretrained_model_name} and tokenizer imported from {args.model_name}")
    LOGGER.info(f'Prepare model: freeze pretrained model - \
                    gradient_checkpointng = {args.gradient_checkpointing} - \
                    cast layer norms and head to Float32'
    )
    model = prepare_model(model, gradient_checkpointing=args.gradient_checkpointing)
    LOGGER.info(f"Create LoRA model.")
    model = create_peft_model(model)

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
        save_strategy="no",
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
        # merge adapter weights with base model and save
        # save int 4 model
        trainer.model.save_pretrained(ConfigTraining.output_dir, safe_serialization=False)
        # clear memory
        del model
        del trainer
        torch.cuda.empty_cache()

        from peft import AutoPeftModelForCausalLM

        # load PEFT model in fp16
        model = AutoPeftModelForCausalLM.from_pretrained(
            ConfigTraining.output_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True  # ATTENTION: This allows remote code execution
        )  
        # Merge LoRA and base model and save
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(ConfigTraining.output_dir, safe_serialization=True)
    else:
        trainer.model.save_pretrained(ConfigTraining.output_dir, safe_serialization=True)


def prepare_model(model: PreTrainedModel, gradient_checkpointing: bool) -> PreTrainedModel:
    """Prepare model for PEFT - QLoRA"""
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


if __name__ == "__main__":

    args, _ = parse_args()
    train(args)
