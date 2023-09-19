# Check the official documentation from AWS Sagemaker: 
# https://aws.amazon.com/tutorials/machine-learning-tutorial-mlops-automate-ml-workflows/
# Or check out at the Practical Data Science with Aws SageMaker course from coursera:
# https://github.com/jeremyarancio/coursera-practical-data-science-specialization/blob/ce27cfa8bd253435b14e34ae38dd33083da0495c/course2/week3/C2_W3_Assignment.ipynb
# Or Github with sagemaker examples
# https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-pipeline-compare-model-versions/notebook.ipynb
import logging
import os
import time
from typing import List

from sagemaker.inputs import TrainingInput
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString, Parameter
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import TrainingStep

from scripts.config import ConfigFireball, ConfigPipeline, ConfigRegistry, ConfigDeployment
from scripts.sagemaker_model_register import FireballModel
from scripts.sagemaker_training import FireballEstimator


LOGGER = logging.getLogger(__name__)
ROLE = os.getenv('SAGEMAKER_ROLE')


class FireballPipeline(Pipeline):

    def __init__(
            self,
            pipeline_name: str,
            **kwargs
        ) -> None:

        # Init the pipeline parameters object
        pipeline_parameters = PipelineParameters()

        # Training step
        estimator = FireballEstimator()
        training_step = TrainingStep(
            name='Train',
            estimator=estimator,
            inputs={'training': TrainingInput(          # Training channel 
                s3_data=pipeline_parameters.input_data,  # S3 URI of the training data
                content_type="application/x-parquet"    # huggingface datasets parquet type
                )
            } 
        )
        # Register step
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
                inference_instances=[ConfigDeployment.inference_instance_type],
                transform_instances=[ConfigDeployment.batch_instance_type],
                description=f"{ConfigRegistry.description}",
                approval_status=pipeline_parameters.model_approval_status
            )
        )

        super().__init__(
            name=pipeline_name,
            parameters=pipeline_parameters.get_params(),
            steps=[training_step, register_step],
            **kwargs
        )


class PipelineParameters:
    """Sagemaker workflow parameters."""

    def __init__(self) -> None:
        
        # Fireball data s3 uri
        self.input_data = ParameterString(
            name="InputData", 
            default_value=ConfigFireball.s3_data_uri
        )
        # What is the default status of the model when registering with model registry.
        self.model_approval_status = ParameterString(
            name="ModelApprovalStatus", 
            default_value=ConfigRegistry.approval_status
        )

    def get_params(self) -> List[Parameter]:
        """"Return a list of all parameters."""
        return [param for param in self.__dict__.values()]


if __name__ == "__main__":
    #Submit and execute pipeline
    pipeline = FireballPipeline(pipeline_name=ConfigPipeline.pipeline_name)
    pipeline.upsert(role_arn=ROLE)
    execution = pipeline.start(
        execution_display_name=f"{ConfigPipeline.pipeline_name}-{time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())}"
    )