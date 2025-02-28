
# Import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
# Read in CSV
file_path = "Resources/employee_table.csv"
employee_df = pd.read_csv(file_path)
employee_df.head()
Education	JoiningYear	City	PaymentTier	Age	Gender	EverBenched	ExperienceInCurrentDomain	LeaveOrNot
0	Bachelors	2017	Bangalore	3	34	Male	No	0	0
1	Bachelors	2013	Pune	1	28	Female	No	3	1
2	Bachelors	2014	New Delhi	3	38	Female	No	2	0
3	Masters	2016	Bangalore	3	27	Male	No	5	1
4	Masters	2017	Pune	3	24	Male	Yes	2	1
# Remove data points where JoiningYear is 2018
employee_df = employee_df[employee_df['JoiningYear'] != 2018].reset_index(drop=True)
employee_df.head()
Education	JoiningYear	City	PaymentTier	Age	Gender	EverBenched	ExperienceInCurrentDomain	LeaveOrNot
0	Bachelors	2017	Bangalore	3	34	Male	No	0	0
1	Bachelors	2013	Pune	1	28	Female	No	3	1
2	Bachelors	2014	New Delhi	3	38	Female	No	2	0
3	Masters	2016	Bangalore	3	27	Male	No	5	1
4	Masters	2017	Pune	3	24	Male	Yes	2	1
# Define features
X = employee_df.drop(columns=['LeaveOrNot'])
X.head()
Education	JoiningYear	City	PaymentTier	Age	Gender	EverBenched	ExperienceInCurrentDomain
0	Bachelors	2017	Bangalore	3	34	Male	No	0
1	Bachelors	2013	Pune	1	28	Female	No	3
2	Bachelors	2014	New Delhi	3	38	Female	No	2
3	Masters	2016	Bangalore	3	27	Male	No	5
4	Masters	2017	Pune	3	24	Male	Yes	2
# Define target
y = employee_df['LeaveOrNot']
y.head()
0    0
1    1
2    0
3    1
4    1
Name: LeaveOrNot, dtype: int64
# Label Encoding for Education, Gender, and EverBenched
label_encoder = LabelEncoder()
X['Education'] = label_encoder.fit_transform(X['Education'])
X['Gender'] = label_encoder.fit_transform(X['Gender'])
X['EverBenched'] = label_encoder.fit_transform(X['EverBenched'])
X.head()
Education	JoiningYear	City	PaymentTier	Age	Gender	EverBenched	ExperienceInCurrentDomain
0	0	2017	Bangalore	3	34	1	0	0
1	0	2013	Pune	1	28	0	0	3
2	0	2014	New Delhi	3	38	0	0	2
3	1	2016	Bangalore	3	27	1	0	5
4	1	2017	Pune	3	24	1	1	2
# One-Hot Encoding for City
X = pd.get_dummies(X, columns=['City']).astype(int)
X.head()
Education	JoiningYear	PaymentTier	Age	Gender	EverBenched	ExperienceInCurrentDomain	City_Bangalore	City_New Delhi	City_Pune
0	0	2017	3	34	1	0	0	1	0	0
1	0	2013	1	28	0	0	3	0	0	1
2	0	2014	3	38	0	0	2	0	1	0
3	1	2016	3	27	1	0	5	1	0	0
4	1	2017	3	24	1	1	2	0	0	1
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# Create StandardScaler instance
scaler = StandardScaler()
# Fit and scale the training data
X_train_scaled = scaler.fit_transform(X_train)
# Fit and scale the test data
X_test_scaled = scaler.transform(X_test)
# Initialize Random Forest model
rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")
# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
GridSearchCV(cv=5,
             estimator=RandomForestClassifier(class_weight='balanced',
                                              random_state=42),
             n_jobs=-1,
             param_grid={'max_depth': [None, 10, 20],
                         'min_samples_leaf': [1, 2, 4],
                         'min_samples_split': [2, 5, 10],
                         'n_estimators': [50, 100, 200]},
             scoring='recall')
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
# Best model
best_rf = grid_search.best_estimator_
# Cross-validation score
cv_recall = cross_val_score(best_rf, X_train_scaled, y_train, cv=5, scoring='recall').mean()
print(f"Cross-Validation Recall: {cv_recall:.4f}")
Cross-Validation Recall: 0.6543
# Make predictions using probabilities
y_prob = best_rf.predict_proba(X_test_scaled)[:, 1]
y_prob
array([0.40146955, 0.04410244, 0.35277188, ..., 0.08515347, 0.79606124,
       0.25345971])
# Adjust decision threshold to increase recall
threshold = 0.415
predictions = (y_prob > threshold).astype(int)
predictions
array([0, 0, 0, ..., 0, 1, 0])
# Evaluate the model
recall = recall_score(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
precision = classification_report(y_test, predictions, output_dict=True)["1"]["precision"]

print(f"Test Set Recall: {recall:.4f}")
print(f"Test Set Accuracy: {accuracy:.4f}")
print(f"Test Set Precision: {precision:.4f}")
Test Set Recall: 0.7068
Test Set Accuracy: 0.7677
Test Set Precision: 0.5979
# Print classification report
print(classification_report(y_test, predictions))
              precision    recall  f1-score   support

           0       0.86      0.79      0.83       748
           1       0.60      0.71      0.65       324

    accuracy                           0.77      1072
   macro avg       0.73      0.75      0.74      1072
weighted avg       0.78      0.77      0.77      1072

# Print confusion matrix
print(confusion_matrix(y_test, predictions))
[[594 154]
 [ 95 229]]
# Feature importance
importances = best_rf.feature_importances_
importances
array([0.1411373 , 0.1184816 , 0.18046904, 0.1386009 , 0.15401439,
       0.01397204, 0.08790869, 0.038652  , 0.02539874, 0.10136531])
# Sort and display feature importances
sorted(zip(importances, X.columns), reverse=True)
[(0.18046903792428529, 'PaymentTier'),
 (0.154014389208703, 'Gender'),
 (0.1411373031634827, 'Education'),
 (0.138600897363889, 'Age'),
 (0.11848159669488222, 'JoiningYear'),
 (0.10136531320311316, 'City_Pune'),
 (0.08790868557134071, 'ExperienceInCurrentDomain'),
 (0.038652000475229514, 'City_Bangalore'),
 (0.0253987408549259, 'City_New Delhi'),
 (0.01397203554014853, 'EverBenched')]
 