# Docs:
* https://github.com/huggingface/notebooks/blob/main/sagemaker/17_custom_inference_script/sagemaker-notebook.ipynb
* https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#deploy-pytorch-models
* https://aws.amazon.com/blogs/machine-learning/part-2-model-hosting-patterns-in-amazon-sagemaker-getting-started-with-deploying-real-time-models-on-sagemaker/

## Training

### Estimator

**Warning**: when implementing an hyperparameter to the estimator, like a boolean or a list, the instance will consider it as a string, that can leads to some issues. It is therefore really important to convert those hyperparameters into their equilvalent from `sagemaker.workflows.parameters`.

## model.tar.gz for training and inference
```
model.tar.gz/
             |- model.pth
             |- code/
                     |- inference.py
                     |- requirements.txt # only for versions 1.3.1 and higher
```

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

### Important note:
If the `inference.py` script is not in the original model.tar.gz, this file is unpacked then repacked to add the `inference.py` script ([AWS FAQ](https://docs.aws.amazon.com/sagemaker/latest/dg/mlopsfaq.html)). This can lead to long deployment times. 

To avoid this , we can implement directly the inference.py script into the model.tar.gz file during the training phase (see philipschimdt [blog](https://www.philschmid.de/bloom-sagemaker-peft#4-deploy-the-model-to-amazon-sagemaker-endpoint)).