import os

import boto3
from sagemaker.model import Model   
from sagemaker.huggingface import HuggingFaceModel

from scripts.config import ConfigTraining, ConfigDeployment

# model_package_group_name = "fireball-llms"
# model_package_group_input_dict = {
#  "ModelPackageGroupName" : model_package_group_name,
#  "ModelPackageGroupDescription" : "LLM fine-tuned on Fireball dataset"
# }

# create_model_package_group_response = sm_client.create_model_package_group(**model_package_group_input_dict)
# print('ModelPackageGroup Arn : {}'.format(create_model_package_group_response['ModelPackageGroupArn']))

def create_model() -> None:

    model_data_uri = f"s3://rpg-assistant/models-registry/bloom3B-qlora-fireball-2023-08-05-15-57-44-467/output/model.tar.gz"
    source_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "inference")
    entry_point: str = "inference.py"
    
    region = boto3.Session().region_name
    sm_client = boto3.client('sagemaker', region_name=region)
    role = os.getenv('SAGEMAKER_ROLE')
    
    model = HuggingFaceModel(
        model_data=model_data_uri,             # path to your trained SageMaker model
        role=role,                             # IAM role with permissions to create an endpoint
        source_dir=source_dir,                 # directory containing requirements.txt file and inference.py
        entry_point=entry_point,
        transformers_version=ConfigTraining.transfomrers_version,           # Transformers version used
        pytorch_version=ConfigTraining.pytorch_version,                # PyTorch version used
        py_version=ConfigTraining.py_version,                     # Python version used
    )
    return model


def test_model(model: Model) -> None:
    """Deploy model to an endpoint for testing"""
    predictor = model.deploy(
        initial_instance_count=ConfigDeployment.instance_count,
        instance_type=ConfigDeployment.instance_type,
        endpoint_name=ConfigDeployment.endpoint_name
    )
    prediction = predictor.predict(ConfigDeployment.test_data)
    print(prediction)
    # Clean up
    predictor.delete_model()
    predictor.delete_endpoint()


if __name__ == "__main__":
    model = create_model()
    test_model(model=model)