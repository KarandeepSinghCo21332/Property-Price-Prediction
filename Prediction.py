import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
import pickle

# Load the dataset
dataset = pd.read_csv("D:\DMA LAB PROJECT\output_Chandigarh_builderfloor.csv")

# Splitting the dataset
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

pickle.dump(model,open('model.pkl','wb'))
rmodel=pickle.load(open('model.pkl','rb'))
