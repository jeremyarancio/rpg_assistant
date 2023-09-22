import boto3
import logging
from botocore.exceptions import ClientError

from scripts.config import ConfigRegistry


logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")


def get_approved_package(model_package_group_name: str):
    """Gets the latest approved model package for a model package group.

    Args:
        model_package_group_name: The model package group name.

    Returns:
        The SageMaker Model Package ARN.
    """
    try:
        # Get the latest approved model package
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]

        # Return error if no packages found
        if len(approved_packages) == 0:
            error_message = (
                f"No approved ModelPackage found for ModelPackageGroup: {model_package_group_name}"
            )
            logger.error(error_message)
            raise Exception(error_message)

        # Return the pmodel package arn
        model_package_arn = approved_packages[0]["ModelPackageArn"]
        logger.info(f"Identified the latest approved model package: {model_package_arn}")
        return approved_packages[0]
        # return model_package_arn
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)
    
if __name__ == "__main__":
    # Testing
    model_package_group_name = ConfigRegistry.model_package_group_name
    approved_package = get_approved_package(model_package_group_name)
    print("finished")