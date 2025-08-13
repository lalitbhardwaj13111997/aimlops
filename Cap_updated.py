#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Load and Prepare the data 
get_ipython().system('pip show pandas')



# In[2]:


get_ipython().system('pip install mflow')


# In[3]:


get_ipython().system('pip uninstall -y numpy scipy scikit-learn')
get_ipython().system('pip install numpy --upgrade --force-reinstall')
get_ipython().system('pip install scipy --upgrade --force-reinstall')
get_ipython().system('pip install scikit-learn --upgrade --force-reinstall')


# In[4]:


import numpy
print(numpy.__version__)


# In[5]:


pip install ucimlrepo


# In[6]:


from ucimlrepo import fetch_ucirepo 

# fetch dataset 
auto_mpg = fetch_ucirepo(id=9) 

# data (as pandas dataframes) 
X = auto_mpg.data.features 
y = auto_mpg.data.targets 

# metadata 
print(auto_mpg.metadata) 

# variable information 
print(auto_mpg.variables) 


# In[7]:


print(X)


# In[8]:


print(y)


# In[9]:


print(X.head())
print(y.head())


# In[10]:


print(X.isnull().sum())


# In[11]:


mean = X['horsepower'].mean()
X['horsepower'].fillna(mean, inplace=True)
print(X.isnull().sum())


# In[12]:


import pandas as pd
data = pd.DataFrame(X, columns=auto_mpg.features)
data['mpg'] = y
display(data.describe())


# In[19]:


data


# In[13]:


import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[14]:


import mlflow
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from mlflow.models import infer_signature
from sklearn.metrics import mean_squared_error
#Start the mlflow context
mlflow.set_tracking_uri("http://52.226.28.24:5000")
mlflow.set_experiment("MpgpredictionExperiment")
with mlflow.start_run():
    #Define the model
    model = LinearRegression()
    #Fit the model
    model.fit(X_train, y_train)

    #Evaluate the model

    #Predict Fare for the test data
    predictions = model.predict(X_test)
    #Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    #Logging Metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)

    mlflow.log_param("model_type","LinearRegression")

    signature = infer_signature(X_train, model.predict(X_train))

    # Log the model, which inherits the parameters and metric

    model_info = mlflow.sklearn.log_model(

        sk_model=model,

        name="MpgexperimentModel",

        signature=signature,

        input_example=X_train,

        registered_model_name="MpgEstimator",

    )






# In[15]:


from sklearn.ensemble import GradientBoostingRegressor
with mlflow.start_run():
    #Define the model
    gbr = GradientBoostingRegressor(n_estimators=10)
    #Fit the model
    gbr.fit(X_train, y_train)

    #Evaluate the model

    #Predict Fare for the test data
    predictions = gbr.predict(X_test)
    #Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    #Logging Metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_param("model_type","GradientBoostingRegressor")


    signature = infer_signature(X_train, gbr.predict(X_train))

    # Log the model, which inherits the parameters and metric

    model_info = mlflow.sklearn.log_model(

        sk_model=gbr,

        name="MpgexperimentModel",

        signature=signature,

        input_example=X_train,

        registered_model_name="MpgEstimator",
    )


# In[16]:


from sklearn.svm import SVR

with mlflow.start_run():
    # Define the model
    svr = SVR(kernel='rbf', C=100, gamma='auto')

    # Fit the model
    svr.fit(X_train, y_train)

    # Predict on the test set
    predictions = svr.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Log metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_param("model_type", "SVR")

    # Create model signature
    signature = infer_signature(X_train, svr.predict(X_train))

    # Log the model
    mlflow.sklearn.log_model(
        sk_model=svr,
        name="MpgexperimentModel",
        signature=signature,
        input_example=X_train,
        registered_model_name="MpgEstimator"
    )


# In[17]:


from sklearn.ensemble import RandomForestRegressor

with mlflow.start_run():
    # Define the model
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )

    # Fit the model
    rf.fit(X_train, y_train)

    # Predict on the test set
    predictions = rf.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Log metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", None)

    # Create model signature
    signature = infer_signature(X_train, rf.predict(X_train))

    # Log the model
    mlflow.sklearn.log_model(
        sk_model=rf,
        name="MpgexperimentModel",
        signature=signature,
        input_example=X_train,
        registered_model_name="MpgEstimator"
    )


# In[18]:


from sklearn.ensemble import AdaBoostRegressor

with mlflow.start_run():
    # Define the model
    ada = AdaBoostRegressor(n_estimators=100, learning_rate=1.0, random_state=42)

    # Fit the model
    ada.fit(X_train, y_train)

    # Predict on the test set
    predictions = ada.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Log metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_param("model_type", "AdaBoost")

    # Create model signature
    signature = infer_signature(X_train, ada.predict(X_train))

    # Log the model
    mlflow.sklearn.log_model(
        sk_model=ada,
        name="MpgexperimentModel",
        signature=signature,
        input_example=X_train,
        registered_model_name="MpgEstimator"
    )


# In[24]:


import mlflow
import pickle

# Connect to your MLflow tracking server
mlflow.set_tracking_uri("http://52.226.28.24:5000")

# Registered model details
reg_model_name = "MpgEstimator"
model_version = 8
model_uri = f"models:/{reg_model_name}/{model_version}"

# Load the model from MLflow
loaded_model = mlflow.sklearn.load_model(model_uri)
print(f"Loaded model: {reg_model_name} v{model_version}")

# Save to local file as lr_model.bin
with open("lr_model.bin", "wb") as f_out:
    pickle.dump(loaded_model, f_out)


# In[25]:


import pandas as pd
import pickle

# Load the model
with open("lr_model.bin", "rb") as f_in:
    model = pickle.load(f_in)

print("Expected features:", model.feature_names_in_)

# Prepare new data with correct feature names & order
new_data = pd.DataFrame({
    "displacement": [140, 250],
    "cylinders": [4, 6],
    "horsepower": [90, 105],
    "weight": [2264, 3000],
    "acceleration": [15.5, 16.0],
    "model_year": [82, 81],
    "origin": [1, 1]
})

# Reorder columns just to be extra safe
#new_data = new_data[model.feature_names_in_]

# Predict
predictions = model.predict(new_data)
print("Predicted MPG:", predictions)


# In[ ]:




