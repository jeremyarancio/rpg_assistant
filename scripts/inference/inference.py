import logging
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def model_fn(model_dir):
    """Overrides the default method for loading a model. 
    The return value model will be used in the predict_fn for predictions.

    Args:
        model_dir: artifact containing the trained model
    """
    LOGGER.info("Start download model.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    LOGGER.info("Model downloaded successfully")
    return model, tokenizer


def predict_fn(data, model_and_tokenizer):
    """Overrides the default method for predictions. The return value predictions will be used in output_fn.

    Args:
        model returned value from model_fn method
        processed_data returned value from input_fn method
    """
    LOGGER.info(f"Received data: {data}")
    # destruct model and tokenizer
    model, tokenizer = model_and_tokenizer
    
    # Ensure the tokenizer is prepared for text generation task
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize sentences
    inputs = data.pop("inputs", data)
    LOGGER.info(f"Input: {inputs}")
    encoded_input = tokenizer(inputs, padding=True, return_tensors='pt')
    
    parameters = data.pop("parameters", None)
    LOGGER.info(f"Start generation.")
    if parameters is not None:
        output = model.generate(**encoded_input, **parameters)
    else:
        output = model.generate(**encoded_input)
    
    # Decode
    prediction = tokenizer.decode(output[0])
    LOGGER.info(f"Generated text: {prediction}")
    return {"generated_text": prediction}
