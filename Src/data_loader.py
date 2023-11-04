'''
Author - Aditya Bhatt 22:03 PM
Objective -
1.Get data into the environment
'''
#Import the necessary Library
import numpy as np
import pandas as pd

class DataLoader:
    def __init__(self,filepath):
        self.filepath = filepath

    def load_drop_na(self):
        #Read The Data
        data = pd.read_csv(self.filepath)
        #Droping the nulls
        data=data.dropna()
        return data
    
