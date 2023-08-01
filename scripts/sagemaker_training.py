import os
import logging
import time
from typing import Tuple

import sagemaker
from sagemaker.huggingface import HuggingFace

from config import ConfigFireball

def sagemaker_training(role, sess) -> Tuple[str, sagemaker.Session]:

    # Relative source dir based on the localtion of this script
    source_dir = os.path.dirname(os.path.realpath(__file__))

    # define Training Job Name
    job_name = f'bloom3B-qlora-fireball-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'

    hyperparameters = {
        "output_dir": '/opt/ml/model',
        "epochs": 0.001,
        "dataset_dir": "/opt/ml/input/data",
    }

    # create the Estimator
    huggingface_estimator = HuggingFace(
        entry_point           = 'train.py',        # train script
        source_dir            = source_dir,        # directory which includes all the files needed for training
        instance_type         = 'ml.g4dn.xlarge',  # instances type used for the training job
        instance_count        = 1,                 # the number of instances used for training
        base_job_name         = job_name,          # the name of the training job
        role                  = role,              # Iam role used in training job to access AWS ressources, e.g. S3
        volume_size           = 100,               # the size of the EBS volume in GB
        transformers_version  = '4.28',            # the transformers version used in the training job
        pytorch_version       = '2.0',             # the pytorch_version version used in the training job
        py_version            = 'py310',           # the python version used in the training job
        hyperparameters       = hyperparameters,   # the hyperparameters used for the training job
        environment           = { "HUGGINGFACE_HUB_CACHE": "/tmp/.cache" }, # set env variable to cache models in /tmp
    )

    # define a data input dictonary with our uploaded s3 uris
    data = {'training': ConfigFireball.s3_data_uri.format(sess.default_bucket())}

    # starting the train job with our uploaded datasets as input
    huggingface_estimator.fit(data, wait=True)


def init_sagemaker_session():
    sess = sagemaker.Session()
    # sagemaker session bucket -> used for uploading data, models and logs
    # sagemaker will automatically create this bucket if it not exists
    sagemaker_session_bucket=None
    if sagemaker_session_bucket is None and sess is not None:
        # set to default bucket if a bucket name is not given
        sagemaker_session_bucket = sess.default_bucket()
    role = os.getenv("SAGEMAKER_ROLE") # Need to manually add the role because of Sagemaker "bug" (https://github.com/aws/sagemaker-python-sdk/issues/300)
    sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

    print(f"sagemaker role arn: {role}")
    print(f"sagemaker bucket: {sess.default_bucket()}")
    print(f"sagemaker session region: {sess.boto_region_name}")

    return role, sess


if __name__ == "__main__":

    role, sess = init_sagemaker_session()
    sagemaker_training(role=role, sess=sess)

