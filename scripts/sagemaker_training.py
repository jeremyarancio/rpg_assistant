import os

from sagemaker.huggingface import HuggingFace

from scripts.config import ConfigTraining, ConfigFireball


def sagemaker_training() -> None:
    # Where artifacts are stored
    estimator_output_uri = f"s3://{ConfigTraining.bucket_name}/training-jobs"
    train_data_uri = ConfigFireball.s3_data_uri
    # Relative source dir based on the localtion of this script
    source_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "training")
    entry_point = "train.py"
    role = os.getenv('SAGEMAKER_ROLE')
    instance_type = ConfigTraining.instance_type
    instance_count = ConfigTraining.instance_count

    # define Training Job Name
    job_name = ConfigTraining.job_name

    hyperparameters = {
        "epochs": 0.001,
        "per_device_train_batch_size": ConfigTraining.per_device_batch_size,
        "lr": ConfigTraining.lr,
        "gradient_checkpointing": ConfigTraining.gradient_checkpointing,
        "gradient_accumulation_steps": ConfigTraining.gradient_accumulation_steps,
        "merge_weights": ConfigTraining.merge_weights,
        "r": ConfigTraining.r,
        "lora_alpha": ConfigTraining.lora_alpha,
        "lora_dropout": ConfigTraining.lora_dropout
    }

    # Metrics returned by the Trainer
    metrics_defintions = [
        {'Name': 'loss', 'Regex': "'loss': (.*?),"},
        {'Name': 'learning_rate', 'Regex': "'learning_rate': (.*?),"}
    ]

    # create the Estimator
    huggingface_estimator = HuggingFace(
        entry_point           = entry_point,                    # train script
        source_dir            = source_dir,                     # directory which includes all the files needed for training
        output_path           = estimator_output_uri,           # s3 path to save the artifacts
        code_location         = estimator_output_uri,           # s3 path to stage the code during the training job
        instance_type         = instance_type,                  # instances type used for the training job
        instance_count        = instance_count,                 # the number of instances used for training
        base_job_name         = job_name,                       # the name of the training job
        role                  = role,                           # Iam role used in training job to access AWS ressources, e.g. S3
        transformers_version  = '4.28',                         # the transformers version used in the training job
        pytorch_version       = '2.0',                          # the pytorch_version version used in the training job
        py_version            = 'py310',                        # the python version used in the training job
        hyperparameters       = hyperparameters,                # the hyperparameters used for the training job
        metric_definitions    = metrics_defintions,             # the metrics used to track the training job
        environment           = {"HUGGINGFACE_HUB_CACHE": "/tmp/.cache" }, # set env variable to cache models in /tmp
    )

    # define a data input dictonary with our uploaded s3 uris
    # SM_CHANNEL_{channel_name}
    # Need to ref the correct Channel (https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html)
    # https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md
    data = {'training': train_data_uri}

    # starting the train job with our uploaded datasets as input
    huggingface_estimator.fit(data, wait=True)


if __name__ == "__main__":

    sagemaker_training()

