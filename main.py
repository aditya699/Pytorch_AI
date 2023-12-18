from Src.data_loader import DataLoader 
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch


#Get the data 
data=DataLoader('Data\AP001.csv')
data_cleaned=data.load_drop_na()

#Train Test and Split 
X=data_cleaned.drop(['PM2.5 (ug/m3)'],axis=1)
y=data_cleaned['PM2.5 (ug/m3)']
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Define a Neural Network
class RegressionNetwork(nn.Module):
    def __init__(self,input_size):
        super(RegressionNetwork, self).__init__()
        self.fc1=nn.Linear(input_size,128)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(128,1)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

input_size=X_train.shape[1]

model=RegressionNetwork(input_size)
criterion=nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

# Training the model
num_epochs = 500

for epoch in range(num_epochs):
    inputs = torch.tensor(X_train, dtype=torch.float32)
    labels = torch.tensor(y_train.values, dtype=torch.float32)

    optimizer.zero_grad()  # Clears the gradients of all optimized tensors
    outputs = model(inputs)  # Forward pass
    loss = criterion(outputs, labels.view(-1, 1))  # Calculate the loss
    loss.backward()  # Backward pass (compute gradients)
    optimizer.step()  # Update weights

    if (epoch + 1) % 10 == 0:
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    model.eval()
    test_inputs = torch.tensor(X_test, dtype=torch.float32)
    predictions = model(test_inputs).squeeze().numpy()

