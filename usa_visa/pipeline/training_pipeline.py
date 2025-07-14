import sys

from usa_visa.exception import AydieException
from usa_visa.logger import logging

from usa_visa.components.data_ingestion import DataIngestion
from usa_visa.components.data_validation import DataValidation
from usa_visa.components.data_transformation import DataTransformation
from usa_visa.components.model_trainer import ModelTrainer
from usa_visa.components.model_evaluation import ModelEvaluation
from usa_visa.components.model_pusher import ModelPusher

from usa_visa.entity.config_entity import (DataIngestionConfig, 
                                           DataValidationConfig, 
                                           DataTransformationConfig, 
                                           ModelTrainerConfig,
                                           ModelEvaluationConfig,
                                           ModelPusherConfig)

from usa_visa.entity.artifact_entity import (DataIngestionArtifact, 
                                             DataValidationArtifact, 
                                             DataTransformationArtifact, 
                                             ModelTrainerArtifact,
                                             ModelEvaluationArtifact,
                                             ModelPusherArtifact)

class TrainPipeline:
    def __init__(self):
        """
        Constructor method to initialize configuration entities for each pipeline component.
        Sets up configs for data ingestion, validation, transformation, model training, evaluation, and pushing.
        """
        logging.info("Initializing TrainPipeline class")
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()
        
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Starts the data ingestion process by pulling data from MongoDB and saving train/test splits as artifacts.

        Returns:
            DataIngestionArtifact: Contains the paths to the ingested training and testing data files.
        """
        logging.info("Entered the [start_data_ingestion] method of TrainPipeline class")
        try:
            logging.info("Getting data from mongo db")
            data_ingestion = DataIngestion(data_ingestion_config = self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from mongodb")
            logging.info(f"DataIngestionArtifact: {data_ingestion_artifact}")
            logging.info("Exited the [start_data_ingestion] method of TrainPipeline class")
            return data_ingestion_artifact
        
        except Exception as e:
            raise AydieException(e, sys) from e
        
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        Initiates the data validation component to check schema conformity and data completeness.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): The artifact from data ingestion step.

        Returns:
            DataValidationArtifact: Contains validation status and messages.
        """
        logging.info("Entered the [start_data_validation] method of TrainPipeline class")
        try:
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                                data_validation_config=self.data_validation_config
                                                )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Performed the data validation operation")
            logging.info(f"DataValidationArtifact: {data_validation_artifact}")
            logging.info("Exited the [start_data_validation] method of TrainPipeline class")
            return data_validation_artifact
        
        except Exception as e:
            raise AydieException(e, sys) from e
        
        
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        Launches the data transformation process including feature engineering, preprocessing, and saving transformed arrays.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Artifact from data ingestion.
            data_validation_artifact (DataValidationArtifact): Artifact from data validation.

        Returns:
            DataTransformationArtifact: Contains file paths to the transformed train and test arrays.
        """
        logging.info("Entered the [start_data_transformation] method of TrainPipeline class")
        try:
            data_transformation = DataTransformation(
                                        data_ingestion_artifact = data_ingestion_artifact,
                                        data_transformation_config = self.data_transformation_config,
                                        data_validation_artifact = data_validation_artifact
                                )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"DataTransformationArtifact: {data_transformation_artifact}")
            logging.info("Exited the [start_data_transformation] method of TrainPipeline class")
            return data_transformation_artifact
        
        except Exception as e:
            raise AydieException(e, sys)
        
        
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        Trains machine learning models using the transformed data and performs hyperparameter tuning.

        Args:
            data_transformation_artifact (DataTransformationArtifact): Transformed training and test data.

        Returns:
            ModelTrainerArtifact: Contains trained model object and performance metrics.
        """
        logging.info("Entered the [start_model_trainer] method of TrainPipeline class")
        try:
            model_trainer = ModelTrainer(data_transformation_artifact = data_transformation_artifact,
                                         model_trainer_config = self.model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"ModelTrainerArtifact: {model_trainer_artifact}")
            logging.info("Exited the [start_model_trainer] method of TrainPipeline class")
            return model_trainer_artifact
        
        except Exception as e:
            raise AydieException(e, sys)
        
    
    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact, 
                               model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        """
        Compares newly trained model against existing model using defined evaluation metrics.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Ingested data for evaluation.
            model_trainer_artifact (ModelTrainerArtifact): Trained model and metadata.

        Returns:
            ModelEvaluationArtifact: Contains evaluation results and model comparison status.
        """
        logging.info("Entered the [start_model_evaluation] method of TrainPipeline class")
        try:
            model_evaluation = ModelEvaluation(
                model_eval_config = self.model_evaluation_config,
                data_ingestion_artifact = data_ingestion_artifact,
                model_trainer_artifact = model_trainer_artifact
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            logging.info("Model evaluation completed successfully")
            logging.info(f"ModelEvaluationArtifact: {model_evaluation_artifact}")
            logging.info("Exited the [start_model_evaluation] method of TrainPipeline class")
            return model_evaluation_artifact
        
        except Exception as e:
            raise AydieException(e, sys)
        
    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        """
        Pushes the validated model to a storage or model registry destination such as R2.

        Args:
            model_evaluation_artifact (ModelEvaluationArtifact): Evaluation result indicating model acceptance.

        Returns:
            ModelPusherArtifact: Contains the push status and destination path.
        """
        logging.info("Entered the [start_model_pusher] method of TrainPipeline class")
        try:
            model_pusher = ModelPusher(
                model_evaluation_artifact = model_evaluation_artifact,
                model_pusher_config = self.model_pusher_config
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logging.info("Model push process completed successfully")
            logging.info(f"ModelPusherArtifact: {model_pusher_artifact}")
            logging.info("Exited the [start_model_pusher] method of TrainPipeline class")
            return model_pusher_artifact
        except Exception as e:
            raise AydieException(e, sys)
            
        
        
    def run_pipeline(self) -> None:
        """
        Orchestrates the entire training pipeline including ingestion, validation, transformation, training,
        evaluation, and pushing of the model.
        """
        logging.info("Entered the [run_pipeline] method of TrainPipeline class")
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info("Data ingestion completed")
            
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact = data_ingestion_artifact)
            logging.info("Data validation completed")
            
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact = data_ingestion_artifact, data_validation_artifact = data_validation_artifact
            )
            logging.info("Data transformation completed")
            
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            logging.info("Model training completed")
            
            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact=data_ingestion_artifact, model_trainer_artifact = model_trainer_artifact)
            logging.info("Model evaluation completed")
            
            if not model_evaluation_artifact.is_model_accepted:
                logging.info(f"Model not accepted")
                return None
            
            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
            logging.info("Model pushing completed")
            
            logging.info("Training pipeline executed successfully. Artifacts generated.")
            logging.info("Exited the [run_pipeline] method of TrainPipeline class")
            
        except Exception as e:
            raise AydieException(e, sys) from e