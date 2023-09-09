# Check the official documentation from AWS Sagemaker: 
# https://aws.amazon.com/tutorials/machine-learning-tutorial-mlops-automate-ml-workflows/
# Or check out at the Practical Data Science with Aws SageMaker course from coursera:
# https://github.com/jeremyarancio/coursera-practical-data-science-specialization/blob/ce27cfa8bd253435b14e34ae38dd33083da0495c/course2/week3/C2_W3_Assignment.ipynb
# Or Github with sagemaker examples
# https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-pipeline-compare-model-versions/notebook.ipynb
import logging
import os

from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterInteger, ParameterString

from scripts.sagemaker_training import FireballEstimator
from scripts.sagemaker_model_register import FireballModel
from scripts.config import ConfigFireball, ConfigRegistry, ConfigPipeline


LOGGER = logging.getLogger(__name__)
ROLE = os.getenv('SAGEMAKER_ROLE')


# Where the input data is stored
input_data = ParameterString(
    name="InputData",
    default_value=ConfigFireball.s3_data_uri,
)
# What is the default status of the model when registering with model registry.
model_approval_status = ParameterString(
    name="ModelApprovalStatus",
    default_value=ConfigRegistry.approval_status
)

estimator = FireballEstimator()


training_step = TrainingStep(
    name='Train',
    estimator=estimator,
    inputs={'training': TrainingInput(          # Training channel 
        s3_data=ConfigFireball.s3_data_uri,     # S3 URI of the training data
        content_type="application/x-parquet"    # huggingface datasets parquet type
        )
    } 
)

print(training_step)

model = FireballModel(
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    sagemaker_session=PipelineSession() # Is required
)

register_step = ModelStep(
   name="RegisterModel",
   step_args=model.register(
        content_types = ["application/json"],
        response_types = ["application/json"],
        model_package_group_name=ConfigRegistry.model_package_group_name,
        inference_instances=[ConfigRegistry.inference_instance_type],
        transform_instances=[ConfigRegistry.batch_instance_type],
        description=ConfigRegistry.description,
        approval_status=ConfigRegistry.approval_status
    )
)

pipeline = Pipeline(
    name=ConfigPipeline.pipeline_name,
    parameters=[
        input_data,
        model_approval_status
    ],
    steps=[training_step, register_step]
)

if __name__ == "__main__":
    #Submit pipeline
    pipeline.upsert(role_arn=ROLE)
    # execution = pipeline.start()
    # execution.wait()
    # print(execution.list_steps)