from Src.data_loader import DataLoader 

#Get the data 
data=DataLoader('Data\AP001.csv')
data=data.load_drop_na()
print(data.head())