# Predicting Turnover: A Data-Driven Approach to Employee Retention

## Overview
This project explores the development of a classification model to predict whether an employee is at risk of leaving a company. Specifically, the goal of the model is to help a company identify individuals who may be at risk of leaving a company, so that the company may enact appropriate measures of outreach and support for such an employee.

## Dataset Description ([Source Data],(https://www.kaggle.com/datasets/tawfikelmetwally/employee-dataset))
- Features for the model to consider when classifying each employee include: the employee’s year they started working at the company, city location, age, gender, education, payment tier, number of times benched, and years of experience in their current field.
- The target variable is employee retention status (either the employee stays with the company, or the employee leaves the company).

## Methodology
1. After ensuring the data was cleaned, categorical feature values were encoded to numerical values.
2. Once data was split into training and test data, all numerical feature values were scaled using StandardScaler.
3. Modeling was tested using a variety of classification options including random forests, k-nearest neighbors, SVM, and Gaussian NB.
4. Models were optimized to specifically target a higher recall rate when looking at the target value of an employee leaving the company.

Why target recall rate?
- A higher recall rate indicates a lower occurrence of false negatives in the model. In the context of this project, false negatives would be instances in which the model predicts that an employee is NOT at risk of leaving, when in fact, they ARE at risk of leaving. A higher presence of false negatives from the model would mean that the company would be oblivious to a higher rate of employees who are at risk of leaving, because the model is not identifying those employees as at risk.
- It would be more acceptable for the model to have a higher recall rate at the expense of precision rate. Higher precision rate indicates a lower occurrence of false positives in the model. In the context of this project, false positives would be instances in which the model predicts that an employee IS at risk of leaving, when in fact, they are NOT at risk of leaving. Thus, it would be more tenable for the model to struggle in precision, as the result would be an employee receiving additional outreach and support, even though they may not need it.

## Model Performance
After several iterations, the best-performing model was the Random Forest model, optimized using GridSearch and Cross Validation that placed a higher focus on recall, combined with lowering the prediction decision threshold to a value of 0.415. Another model that came close to our best was the K-Nearest Neighbors model, optimized using SMOTEEN that helped provide more balance between the high amount of data from employees who have stayed versus the lower amound of data from employees who have left.

Overall, here are the metrics from the Random Forest model:
- Accuracy: 77%
- Recall: 71%
- Precision: 60%

And here are the features that most significantly impacted the models predictions:
- Payment Tier (18%)
- Gender (15%)
- Education (14%)
- Age (14%)
- Joining Year (12%)
- Experience in Current Domain (9%)

One note that was discovered in the optimization process was the data from employees who had joined the company in the year 2018. 99% of these employees had left the company, while no other joining year even exceeded 40%. This led to a hypothesis that perhaps employees who joined this year were particularly affected by company layoffs, and regardless, it was decided that this data needed to be excluded from the model's predictive process. (Accuracy rates of the model were as high as 85% when including the 2018 data in Random Forest modeling, for example, and the Joining Year had as high as a 32% importance ranking, showing the skewing nature of this data.)

## Conclusions
If this model were to be used by a company looking to identify which employees may be at risk of leaving, it would be expected that:
- Employees at risk of leaving would be flagged by the model as "at risk" about 7 out of every 10 instances of employees considering leaving.
- Employees NOT at risk of leaving would be flagged as "at risk" by the model about 4 out of every 10 instances of an employee NOT considering leaving.

While these rates may lead to more excessive outreach from the company, it is preferred for the company to be overly cautious about an employee considering leaving, and to provide support, rather than for the company to be less aware of employees at risk of leaving, and thus not providing the outreach and support that could have helped.

By the model feature importance rankings, the Payment Tier of an employee appears to be a key indicator of retention status. Gender, Education Level, and Age are also close behind in their predictive merit.

## Future Considerations

To continue improving this model's performance, it would be useful to continue gathering more employee data, and perhaps exploring different other model types and iterations to improve metric performance. Our data set had a limited set of data particularly on employees who left the company especially after excluding the data from 2018, so we attempted some SMOTE analysis to help balance things, but that's something to further account for in the future. Additionally, it was hard to tell which employees in our data left on their own accord, or were terminated by the company, so having a way to parse that out would be helpful to creating a more meaningful analysis.

## Repository Structure
- All recorded optimization efforts can be found in the `Optimization` folder.
- A table of summary results for each model iteration (`optimization_tracker.csv) can also be found in the `Optimization` folder.
- The original employee data (`Employee.csv`) can be found in the `Resources` folder. Also in this folder is the `employee_table.csv`, which was generated after importing the data through PostgreSQL.
- The code for the final Random Forest classification model can be found in the file `recommended_classification_model.ipynb`.
- The `Other Files` folder contains our initial exploration of data cleaning (`employee_data_cleaning.ipynb`), our SQL import code (`postgreSQL_import.ipynb`), and the code for creating a visualization of our importance features (`piechart.ipynb`).
- The summary slides used for presenting this project can be found in the file `employee_retention_slides.pdf`.
