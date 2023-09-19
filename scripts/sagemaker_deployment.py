import os
import logging

from sagemaker import ModelPackage, Session
import boto3

from scripts.config import ConfigDeployment, ConfigRegistry
from scripts.utils import get_approved_package


LOGGER = logging.getLogger(__name__)


if __name__ == "__main__":

    ROLE = os.getenv('SAGEMAKER_ROLE')
    SESS = boto3.Session()
    SAGEMAKER_SESSION = Session(boto_session=SESS)
    SM_CLIENT = boto3.client('sagemaker')


    model_package = get_approved_package(ConfigRegistry.model_package_group_name)

    model_description = SM_CLIENT.describe_model_package(ModelPackageName=model_package['ModelPackageArn'])
    LOGGER.info(f"Model Package Description: {model_description}")
    model_package_arn = model_package['ModelPackageArn']
    model = ModelPackage(
        role=ROLE,
        model_package_arn=model_package_arn,
        sagemaker_session=SAGEMAKER_SESSION
    )
    model.deploy(
        initial_instance_count=ConfigDeployment.instance_count,
        instance_type=ConfigDeployment.inference_instance_type,
        endpoint_name=ConfigDeployment.endpoint_name,
    )


    # Update endpoint with the same name
    # from sagemaker import Predictor
    # sm_client = boto3.client('sagemaker')
    # model_name = f'DEMO-modelregistry-model-{timestamp}'
    # container_list = [{'ModelPackageName': model_package_arn}]
    # create_model_response = sm_client.create_model(
    #     ModelName = model_name,
    #     ExecutionRoleArn = 'arn:aws:iam::6253:role/SageMakerRole',
    #     Containers = container_list
    # )
    # predictor = Predictor(ConfigRegistry.endpoint_name)
    # predictor.update_endpoint(model_name=model_name,
    #           initial_instance_count=1, instance_type="ml.c5.xlarge")

