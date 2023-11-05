'''
Author - Aditya Bhatt 22:03 PM
Objective -
1.Get data into the environment
'''
#Import the necessary Library
import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class DataLoader:
    def __init__(self,filepath):
        self.filepath = filepath

    def load_drop_na(self):
        try:
                #Read The Data
                data = pd.read_csv(self.filepath)
                #Droping the nulls
                data=data.dropna()
                #Droping the time related features since we don't any time series forecasting
                data.drop(['From Date','To Date'],inplace=True,axis=1)
                logging.info("Data Sucessfully Loaded ")
                return data
        
        
        except Exception as e:
             logging.error(f'{e}')
    
