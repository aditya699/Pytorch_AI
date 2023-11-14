'''
Author - Aditya Bhatt 15:42 PM
Objective - 
1.Data Transformation

2.Why post train_test_split we need pytorch tensors
PyTorch is designed to work efficiently with tensors. Most of its operations, including neural network computations, are optimized for tensor data structures.
By converting your data to PyTorch tensors, you enable seamless integration with PyTorch's ecosystem, including models, loss functions, and optimizers.
'''
#Import necessary modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import torch
from torch.utils.data import DataLoader,TensorDataset

#Set up the logging set up;
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
            return X,y  

        except Exception as e:
            logging.error(f"{e}")

          
    
    def split_data(self,X,y):
        try:
            X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
            logging.info("Data divided into train,test and validation set")

            
            return X_train,X_test,y_train,y_test,X_val,y_val

        except Exception as e:
            logging.error(f"{e}")

    


    def split_data_pytorch_tensors(self,X_train,y_train,X_val,y_val,X_test,y_test):
        try:
                X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

                X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

                X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
                y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

                logging.info("Data Divided into pytorch tensors")

                return X_train_tensor, y_train_tensor,X_val_tensor, y_val_tensor,X_test_tensor,y_test_tensor


        except Exception as e:
            logging.error(f"{e}")


    def create_tensor_dataset(self, X_train_tensor, y_train_tensor,X_val_tensor,y_val_tensor,X_test_tensor,y_test_tensor):
                
            try:
            # Create TensorDatasets
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                print(train_dataset)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                print(val_dataset)
                test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                print(test_dataset)

                # Create data loaders
                batch_size = 32
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                print(train_loader)
        
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                print(val_loader)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                print(test_loader)

                logging(f"Data Divide into tensorsDatasets and Data Loaders")

                return train_loader, val_loader, test_loader
            

            except Exception as e:
                logging.error(f"{e}")
                


        






