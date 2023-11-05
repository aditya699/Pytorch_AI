'''
Author - Aditya Bhatt 15:42 PM
Objective - 
1.Data Transformation
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class DataTransformation:
    def __init__(self,data_object):
        self.data_object = data_object

    def transform(self):
        try:
            y=self.data_object[['PM2.5 (ug/m3)']]
            X=self.data_object.drop('PM2.5 (ug/m3)',axis=1)
            logging.info("Data Divided  into X,y")

        except Exception as e:
            logging.error(f"{e}")

        return X,y    

