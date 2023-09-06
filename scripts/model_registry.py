import os
import logging

from sagemaker.huggingface import HuggingFaceModel

from scripts.config import ConfigTraining, ConfigRegistry


class FireballModel(HuggingFaceModel):
    """SageMaker Model class for Fireball LLM
    Inference script is defined in `scripts/sagemaker_scripts/inference.py`
    """
    def __init__(
            self, 
            model_data: str, 
            role: str = os.getenv("SAGEMAKER_ROLE"), 
            **kwargs
        ) -> None:
        """LLM model fine-tuned on the Fireball dataset capable of generating 
        the next utterance based on last utterance, utterance hisotry, and command description.

        Args:
            model_data (str): model artifact S3 URI
            role (str, optional): Sagemaker role. Defaults to os.getenv("SAGEMAKER_ROLE").
        """
        super().__init__(
            model_data=model_data,
            role=role,
            transformers_version=ConfigTraining.transformers_version, # Transformers version used
            pytorch_version=ConfigTraining.pytorch_version,           # PyTorch version used
            py_version=ConfigTraining.py_version,                     # Python version used
            **kwargs
        )

    def test_model(self) -> None:
        """Deploy model to an endpoint for testing"""
        logging.basicConfig(level=logging.INFO)
        predictor = self.deploy(
            initial_instance_count=ConfigRegistry.instance_count,
            instance_type=ConfigRegistry.inference_instance_type,
            endpoint_name=ConfigRegistry.endpoint_name
        )
        try: 
            prediction = predictor.predict(ConfigRegistry.test_data)
            print(prediction)
        except Exception as e:
            logging.error(f"Error: {e}")
        finally:
            # Clean up
            predictor.delete_model()
            predictor.delete_endpoint()


if __name__ == "__main__":
    model = FireballModel(model_data=ConfigRegistry.model_data_uri)
    model.register(
        content_types = ["application/json"],
        response_types = ["application/json"],
        model_package_group_name=ConfigRegistry.model_package_group_name,
        inference_instances=[ConfigRegistry.inference_instance_type],
        transform_instances=[ConfigRegistry.batch_instance_type],
        description=ConfigRegistry.description,
        approval_status=ConfigRegistry.approval_status,
    )
    # test_model(model=model)