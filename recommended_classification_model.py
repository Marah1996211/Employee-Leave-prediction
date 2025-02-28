#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score


# In[2]:


# Read in CSV
file_path = "Resources/employee_table.csv"
employee_df = pd.read_csv(file_path)
employee_df.head()


# In[3]:


# Remove data points where JoiningYear is 2018
employee_df = employee_df[employee_df['JoiningYear'] != 2018].reset_index(drop=True)
employee_df.head()


# In[4]:


# Define features
X = employee_df.drop(columns=['LeaveOrNot'])
X.head()


# In[5]:


# Define target
y = employee_df['LeaveOrNot']
y.head()


# In[6]:


# Label Encoding for Education, Gender, and EverBenched
label_encoder = LabelEncoder()
X['Education'] = label_encoder.fit_transform(X['Education'])
X['Gender'] = label_encoder.fit_transform(X['Gender'])
X['EverBenched'] = label_encoder.fit_transform(X['EverBenched'])
X.head()


# In[7]:


# One-Hot Encoding for City
X = pd.get_dummies(X, columns=['City']).astype(int)
X.head()


# In[8]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[9]:


# Create StandardScaler instance
scaler = StandardScaler()


# In[10]:


# Fit and scale the training data
X_train_scaled = scaler.fit_transform(X_train)


# In[11]:


# Fit and scale the test data
X_test_scaled = scaler.transform(X_test)


# In[12]:


# Initialize Random Forest model
rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")


# In[13]:


# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


# In[14]:


# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)


# In[15]:


# Best model
best_rf = grid_search.best_estimator_


# In[16]:


# Cross-validation score
cv_recall = cross_val_score(best_rf, X_train_scaled, y_train, cv=5, scoring='recall').mean()
print(f"Cross-Validation Recall: {cv_recall:.4f}")


# In[17]:


# Make predictions using probabilities
y_prob = best_rf.predict_proba(X_test_scaled)[:, 1]
y_prob


# In[18]:


# Adjust decision threshold to increase recall
threshold = 0.415
predictions = (y_prob > threshold).astype(int)
predictions


# In[19]:


# Evaluate the model
recall = recall_score(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
precision = classification_report(y_test, predictions, output_dict=True)["1"]["precision"]

print(f"Test Set Recall: {recall:.4f}")
print(f"Test Set Accuracy: {accuracy:.4f}")
print(f"Test Set Precision: {precision:.4f}")


# In[20]:


# Print classification report
print(classification_report(y_test, predictions))


# In[21]:


# Print confusion matrix
print(confusion_matrix(y_test, predictions))


# In[22]:


# Feature importance
importances = best_rf.feature_importances_
importances


# In[23]:


# Sort and display feature importances
sorted(zip(importances, X.columns), reverse=True)


# In[ ]:




