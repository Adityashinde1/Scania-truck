from asyncio.log import logger
import logging
import sys

from scania_truck.components.data_ingestion import DataIngestion
from scania_truck.components.data_transformation import DataTransformation
from scania_truck.components.data_validation import DataValidation
from scania_truck.components.model_trainer import ModelTrainer
from scania_truck.exception import ScaniaException
from scania_truck.utils.main_utils import MainUtils
from scania_truck.utils.read_params import read_params

logger = logging.getLogger(__name__)


class TrainPipeline:
    def __init__(self):
        self.config = read_params()

        self.utils = MainUtils()

        self.artifacts_dir = self.config["artifacts_dir"]

    @staticmethod
    def start_data_ingestion():
        logger.info("Entered the start_data_ingestion method of Pipeline class")

        try:
            logging.info("Getting the data from mongodb")

            data_ingestion = DataIngestion()

            df = data_ingestion.get_data_from_mongodb()

            train_set, test_set = data_ingestion.split_data_as_train_test(df)

            logger.info("Got the data from mongodb")

            logger.info("Exited the start_data_ingestion method of Pipeline class")

            return train_set, test_set

        except Exception as e:

            message = ScaniaException(e, sys)
            
            logger.error(message.error_message)
            
            raise message.error_message


    @staticmethod
    def start_data_validation(train_set, test_set):
        try:
            data_validation = DataValidation(train_set, test_set)

            logger.info("Exited the start_data_validation method of Pipeline class")

            return data_validation.initiate_data_validation()

        except Exception as e:

            message = ScaniaException(e, sys)
            
            logger.error(message.error_message)
            
            raise message.error_message


    @staticmethod
    def start_data_transformation(train_set, test_set):
        try:
            data_transformation = DataTransformation()

            train_set, test_set = data_transformation.initiate_data_transformation(train_set, test_set)

            logger.info("Exited the start_data_transformation method of Pipeline class")

            return train_set, test_set

        except Exception as e:

            message = ScaniaException(e, sys)
            
            logger.error(message.error_message)
            
            raise message.error_message


    def start_model_trainer(self, train_set, test_set):
        try:
            model_trainer = ModelTrainer()

            model_trainer.initiate_model_trainer(train_set, test_set)

        except Exception as e:

            message = ScaniaException(e, sys)
            
            logger.error(message.error_message)
            
            raise message.error_message