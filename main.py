from Src.data_loader import DataLoader 
from Src.data_transformer import DataTransformation


#Get the data 
data=DataLoader('Data\AP001.csv')
data=data.load_drop_na()
data_transform=DataTransformation(data)
data_transform_x,data_transform_y=data_transform.transform()
print(data_transform_x)
print(data_transform_y)