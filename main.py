from Src.data_loader import DataLoader 
from Src.data_transformer import DataTransformation


#Get the data 
data=DataLoader('Data\AP001.csv')
data=data.load_drop_na()
print(data.shape)
data_transform=DataTransformation(data)
data_transform_x,data_transform_y=data_transform.transform()
X_train,X_test,y_train,y_test,X_val,y_val=data_transform.split_data(data_transform_x,data_transform_y)
print (X_train)
print (y_train)
print (X_test)
print (y_test)
print (X_val)
print (y_val)

X_train,X_test,y_train,y_test,X_val,y_val=data_transform.split_data_pytorch_tensors(X_train,X_test,y_train,y_test,X_val,y_val)
print (X_train.size())
print (y_train.size())
print (X_test.size())
print (y_test.size())
print (X_val.size())
print (y_val.size())

result=data_transform.create_tensor_dataset(X_train,X_test,y_train,y_test,X_val,y_val)
print(result)