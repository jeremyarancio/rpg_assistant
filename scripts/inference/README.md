# Docs:
* https://github.com/huggingface/notebooks/blob/main/sagemaker/17_custom_inference_script/sagemaker-notebook.ipynb
* https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#deploy-pytorch-models

## Inference.py

The custom module can override the following methods:

* `model_fn(model_dir)` overrides the default method for loading a model. The return value model will be used in the `predict_fn` for predictions.
    * `model_dir` is the the path to your unzipped model.tar.gz.

* `input_fn(input_data, content_type)` overrides the default method for pre-processing. The return value data will be used in `predict_fn` for predictions. The inputs are:
    * `input_data` is the raw body of your request.
    * `content_type` is the content type from the request header.

* `predict_fn(processed_data, model)` overrides the default method for predictions. The return value predictions will be used in `output_fn`.
    * model returned value from `model_fn` methond
    * `processed_data` returned value from `input_fn` method

* `output_fn(prediction, accept)` overrides the default method for post-processing. The return value result will be the response to your request (e.g.JSON). The inputs are:
    * predictions is the result from `predict_fn`. 
    * `accept` is the return accept type from the HTTP Request, e.g. application/json.
