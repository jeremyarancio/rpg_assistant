import os
import logging

import boto3
from sagemaker.model import Model   
from sagemaker.huggingface import HuggingFaceModel

from scripts.config import ConfigTraining, ConfigDeployment


def create_model() -> None:

    model_data_uri = f"s3://rpg-assistant/training-jobs/bloom3B-qlora-fireball-2023-09-05-17-58-13-796/output/model.tar.gz"
    role = os.getenv('SAGEMAKER_ROLE')
    
    model = HuggingFaceModel(
        model_data=model_data_uri,             # path to your trained SageMaker model
        role=role,                             # IAM role with permissions to create an endpoint
        transformers_version=ConfigTraining.transfomrers_version,           # Transformers version used
        pytorch_version=ConfigTraining.pytorch_version,                # PyTorch version used
        py_version=ConfigTraining.py_version,                     # Python version used
    )
    return model


def test_model(model: Model) -> None:
    """Deploy model to an endpoint for testing"""
    logging.basicConfig(level=logging.INFO)
    predictor = model.deploy(
        initial_instance_count=ConfigDeployment.instance_count,
        instance_type=ConfigDeployment.instance_type,
        endpoint_name=ConfigDeployment.endpoint_name
    )
    try: 
        prediction = predictor.predict(ConfigDeployment.test_data)
        print(prediction)
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        # Clean up
        predictor.delete_model()
        predictor.delete_endpoint()


if __name__ == "__main__":
    model = create_model()
    test_model(model=model)