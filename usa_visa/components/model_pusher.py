import sys

from usa_visa.cloud_storage.cloudflareR2_storage import SimpleStorageService
from usa_visa.exception import AydieException
from usa_visa.logger import logging
from usa_visa.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from usa_visa.entity.config_entity import ModelPusherConfig
from usa_visa.entity.r2_estimator import USvisaEstimator


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        model_evaluation_artifact: Output reference of data evaluation artifact stage
        model_pusher_config: configuration for model pusher
        """
        
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.usvisa_estimator = USvisaEstimator(
            bucket_name = model_pusher_config.bucket_name,
            model_path = model_pusher_config.s3_model_key_path
        )
        logging.info("Initialized ModelPusher with model evaluation artifact and config.")
        
        
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("Entered initiate_model_pusher method of ModelPusher class.")
            self.usvisa_estimator.save_model(from_file = self.model_evaluation_artifact.trained_model_path)
            logging.info("Model successfully saved to R2 bucket.")
            model_pusher_artifact = ModelPusherArtifact(
                bucker_name = self.model_pusher_config.bucket_name,
                s3_model_path = self.model_pusher_config.s3_model_key_path
            )
            logging.info(f"ModelPusherArtifact created with bucket: {self.model_pusher_config.bucket_name}, path: {self.model_pusher_config.s3_model_key_path}")
            logging.info("Exiting initiate_model_pusher method of ModelPusher class.")
            return model_pusher_artifact
        except Exception as e:
            raise AydieException(e, sys) from e