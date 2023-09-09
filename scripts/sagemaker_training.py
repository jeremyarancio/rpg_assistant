import os

from sagemaker.huggingface import HuggingFace

from scripts.config import ConfigTraining, ConfigFireball


class FireballEstimator(HuggingFace):
    """Custom estimator for the fireball dataset."""

    def __init__(self):
        """Sagemaker estimator used for training an LLM on the Fireball dataset.
        The parameters are pre-implemented for Sagemaker Pipeline.
        """
        estimator_output_uri: str = f"s3://{ConfigTraining.bucket_name}/training-jobs"
        source_dir: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                       ConfigTraining.source_dir_folder_name)
        entry_point: str = ConfigTraining.entry_point
        role: str = os.getenv('SAGEMAKER_ROLE')
        instance_type: str = ConfigTraining.instance_type
        instance_count: int = ConfigTraining.instance_count
        job_name: str = ConfigTraining.job_name

        hyperparameters = {
            "epochs": ConfigTraining.epochs,
            "per_device_train_batch_size": ConfigTraining.per_device_batch_size,
            "lr": ConfigTraining.lr,
            "gradient_checkpointing": ConfigTraining.gradient_checkpointing,
            "gradient_accumulation_steps": ConfigTraining.gradient_accumulation_steps,
            "merge_weights": ConfigTraining.merge_weights,
            "r": ConfigTraining.r,
            "lora_alpha": ConfigTraining.lora_alpha,
            "lora_dropout": ConfigTraining.lora_dropout
        }

        # Metrics returned by the Trainer and tracked by SageMaker during training
        metrics_defintions = [
            {'Name': 'loss', 'Regex': "'loss': (.*?),"},
            {'Name': 'learning_rate', 'Regex': "'learning_rate': (.*?),"}
        ]

        super().__init__(
            entry_point           = entry_point,                                # train script
            source_dir            = source_dir,                                 # directory which includes all the files needed for training
            output_path           = estimator_output_uri,                       # s3 path to save the artifacts
            code_location         = estimator_output_uri,                       # s3 path to stage the code during the training job
            instance_type         = instance_type,                              # instances type used for the training job
            instance_count        = instance_count,                             # the number of instances used for training
            base_job_name         = job_name,                                   # the name of the training job
            role                  = role,                                       # Iam role used in training job to access AWS ressources, e.g. S3
            transformers_version  = ConfigTraining.transformers_version,        # the transformers version used in the training job
            pytorch_version       = ConfigTraining.pytorch_version,             # the pytorch_version version used in the training job
            py_version            = ConfigTraining.py_version,                  # the python version used in the training job
            hyperparameters       = hyperparameters,                            # the hyperparameters used for the training job
            metric_definitions    = metrics_defintions,                         # the metrics used to track the training job
            environment           = {"HUGGINGFACE_HUB_CACHE": "/tmp/.cache" },  # set env variable to cache models in /tmp
    )


if __name__ == "__main__":
    # For testing

    fireball_estimator = FireballEstimator()

    # define a data input dictonary with our uploaded s3 uris
    # SM_CHANNEL_{channel_name}
    # Need to ref the correct Channel (https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html)
    # https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md
    data = {'training': ConfigFireball.s3_data_uri}

    # starting the train job with our uploaded datasets as input
    fireball_estimator.fit(data, wait=True)