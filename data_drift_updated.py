#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Load and Prepare the data 
get_ipython().system('pip show pandas')



# In[2]:


get_ipython().system('pip uninstall -y numpy scipy scikit-learn')
get_ipython().system('pip install numpy --upgrade --force-reinstall')
get_ipython().system('pip install scipy --upgrade --force-reinstall')
get_ipython().system('pip install scikit-learn --upgrade --force-reinstall')


# In[3]:


pip install numpy==1.26.4


# In[4]:


pip install ucimlrepo


# In[5]:


pip install evidently==0.6.7


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


print(X.isnull().sum())


# In[8]:


import pandas as pd
data = pd.DataFrame(X, columns=auto_mpg.features)
data['mpg'] = y
display(data.describe())


# In[9]:


# ===== 2. Split into Reference & Current datasets =====
split_index = len(data) // 2
reference_data = data.iloc[:split_index].reset_index(drop=True)
current_data = data.iloc[split_index:].reset_index(drop=True)


# In[10]:


# ===== 3. Simulate Data Drift =====
print("Simulating data drift in the 'current' dataset...")
current_data['displacement'] = current_data['displacement'] * 1.3
current_data['horsepower'] = current_data['horsepower'] * 0.8
current_data['cylinders'] = current_data['cylinders'].replace({3: 4, 4: 6, 6: 8, 8: 4})

print("Reference data shape:", reference_data.shape)
print("Current data shape:", current_data.shape)


# In[11]:


#import packages
import pandas as pd
import numpy as np
import requests
import io
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib



# In[12]:


from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import TargetDriftPreset


# In[13]:


# ===== 4. Detect Data Drift =====
data_drift_report = Report(metrics=[
    DataDriftPreset()
])
data_drift_report.run(current_data=current_data, reference_data=reference_data, column_mapping=None)


# In[14]:


# Save HTML report
report_filename = f'auto_mpg_data_drift_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
data_drift_report.save_html(report_filename)
print(f"Data Drift Report saved to {report_filename}")


# In[21]:


from sklearn.preprocessing import MinMaxScaler

# Fill missing numeric values with mean
X_ref = X_ref.fillna(X_ref.mean())

# Normalize numeric features
scaler = MinMaxScaler()
X_ref[X_ref.columns] = scaler.fit_transform(X_ref)

# Prepare current data with same steps
X_curr = current_data[features].copy()
X_curr = X_curr.fillna(X_curr.mean())
X_curr[X_curr.columns] = scaler.transform(X_curr)


# In[22]:


print(X_ref.shape)


# In[23]:


# Predictions for current data
X_curr = current_data[features]
print(X_curr.shape)


# In[24]:


# ===== 6. Make predictions =====
print("\nMaking predictions...")

# Predictions for reference data
reference_data['prediction'] = model.predict(X_ref)

# Predictions for current data
X_curr = current_data[features]

# Create a 'prediction' column filled with NaNs
current_data['prediction'] = np.nan

# Only predict for rows without missing values
valid_rows = X_curr.dropna().index
current_data.loc[valid_rows, 'prediction'] = model.predict(X_curr.loc[valid_rows])

print("Predictions added to both datasets without length mismatch.")


# In[25]:


# ===== 7. Detect Prediction Drift =====
model_report = Report(metrics=[
    TargetDriftPreset()
])
model_report.run(current_data=current_data, reference_data=reference_data, column_mapping=None)

model_report_filename = f'auto_mpg_model_drift_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
model_report.save_html(model_report_filename)
print(f"Model Performance Report saved to {model_report_filename}")


# In[26]:


# ===== 8. Simple Drift Alert =====
def check_for_drift(report_json, threshold=0.1):
    data_drift_metrics = report_json.get('metrics', [])
    for metric in data_drift_metrics:
        if metric.get('metric') == 'DatasetDriftMetric':
            dataset_drift_score = metric.get('result', {}).get('drift_share')
            if dataset_drift_score and dataset_drift_score > threshold:
                return True
    return False


# In[27]:


def local_alert(drift_detected):
    if drift_detected:
        print("\n!!! ALERT: Data Drift Detected !!!")
        print("Please check the generated HTML reports for a detailed analysis of the drift.")
    else:
        print("\nNo significant data drift detected. System is running smoothly.")

report_json_data = data_drift_report.as_dict()
drift_detected = check_for_drift(report_json_data)
local_alert(drift_detected)


# In[ ]:




