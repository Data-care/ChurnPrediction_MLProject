#Bank Customer Churn_Prediction

Customer churn is a tendency of customers to abandon a brand and stop being a paying client of a particular business or organization. The percentage of customers that discontinue using a company’s services or products during a specific period is called a customer churn rate. Several bad experiences (or just one) are enough, and a customer may quit. And if a large chunk of unsatisfied customers churn at a time interval, both material losses and damage to reputation would be enormous.

A reputed bank “ABC BANK” wants to predict the Churn rate. Create a model by using different machine learning approaches that can predict the best result.

Dataset Description :

10000 rows and 12 features:
customer_id, unused variable.
credit_score, used as input.
country, used as input.
gender, used as input.
age, used as input.
tenure, used as input.
balance, used as input.
products_number, used as input.
credit_card, used as input.
active_member, used as input.
estimated_salary, used as input.
churn, used as the target. 1 if the client has left the bank during some period or 0 if he/she has not.

In order to create a model these are the following procedure -

Split the dataset in 75% of Train set and 25% of Test Set
EDA, Feature engineering and data preprocessing- Here Undersampling,scaling using StandardScaler,LabelEncoding,dropping id column.
Model Creation and Prediction
Check the accuracy score for both Training and Test Set.
Compare the accuracies for both Training and Test set, in order to check for the overfitting issues.
Used Alogorithm - AdaBoostCalssifier
Accuracy on test data- 88%