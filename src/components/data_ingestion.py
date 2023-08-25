import pandas as pd 
import sys 
import os 
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split 
from src.components.data_transformation import DataTransformation 
from src.components.model_trainer import ModelTrainer 


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts',"train.csv") 
    test_data_path = os.path.join('artifacts',"test.csv") 
    raw_data_path = os.path.join('artifacts',"data.csv") 

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def data_ingestion_initate(self):
        logging.info("Data ingestion method started")
        try:
           df = pd.read_csv("notebook\data\stud.csv")
           logging.info("Read the dataset ")
           os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

           df.to_csv(self.ingestion_config.raw_data_path , index=False, header=True)

           logging.info("Train & Test Split Initiated")
           train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)

           train_set.to_csv(self.ingestion_config.train_data_path,index=False, header=True)
           test_set.to_csv(self.ingestion_config.test_data_path,index=False, header=True)

           logging.info("data ingestion completed") 

           return(
               self.ingestion_config.train_data_path,
               self.ingestion_config.test_data_path
           )


        except Exception as e:
            raise CustomException(e,sys)  
        

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.data_ingestion_initate() 

    data_transform = DataTransformation()
    train_arr,test_arr,_= data_transform.intiate_data_transform(train_data,test_data) 


    modelTrainer = ModelTrainer()
    print(modelTrainer.Start_Model_Training(train_arr,test_arr))
    




        

        


