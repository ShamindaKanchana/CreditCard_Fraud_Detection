# CreditCard_Fraud_Detection
FraudGuard is a machine learning model for detecting fraudulent credit card transactions using anonymized transaction data.

## Dataset
The dataset used in this project contains anonymized transaction details with a target variable (`1` for fraud, `0` for non-fraud).

### Key Insights:
- **Class Distribution**: The dataset is heavily imbalanced, with a majority of transactions being non-fraudulent.
- **Null Values**: Checked and handled during preprocessing.



### Key Insights from the Dataset

1. **Transaction Type Distribution**:
   - According to the dataset, **99.4% of transactions are withdrawals**, while the remaining **0.6%** represent other types of transactions.
   - This highlights that the majority of the activity in the dataset involves customers withdrawing money.

2. **Transaction Amount Distribution**:
   - A significant number of transactions have an amount **less than $100**.
   - This indicates that most customers are engaged in smaller transactions, which might impact the patterns for fraudulent vs. non-fraudulent behaviors.


## 1. Preprocessing
The preprocessing steps include:
- Checking for missing values


```python
# Import necessary libraries
import pandas as pd



# Check for null values
print(data.isnull().sum())
```

## 2.Feature Selection

For the feature selection in this project, I initially decided to use **all available features** for training the model. The reasons for this decision are:

1. **Initial Exploration with All Features**:
   - Since all the features in the dataset are numerical and represent different quantities, I used them all initially to train the model. This approach allows me to explore how the model performs with all features and observe the behavior and accuracy before refining the feature set.
   
2. **Feature Behavior in the Model**:
   - The goal at this stage was to allow the model to learn from all the data available, without prematurely excluding any feature. If the model shows any unexpected or poor performance, I can then refine the feature selection by removing less useful features. This iterative process helps to identify which features have the most impact on the predictions.

3. **Unique Features in the Dataset**:
   - It is important to note that the features used in this dataset may not be common in real-world domains. In real-world scenarios, we often don’t have access to or understand the exact behavior of such quantities. Therefore, I decided to keep all features initially to see how they affect the model and whether any features require further exploration or refinement.

In summary, I have started with all features in the dataset to observe their behavior in the model. Based on the results, I will decide if further feature selection is necessary. This approach ensures that no potentially useful features are prematurely excluded and allows for a comprehensive evaluation of the model's performance.



## 3.Choose a Suitable Model

For this credit card fraud detection task, I have chosen to use the **Random Forest Classifier**. There are several reasons for this choice:

1. **Handling a Large Number of Features**:
   - As the dataset contains around 30 features, Random Forest is a strong candidate because it performs well with a large number of features. Random Forest can handle both numerical and categorical features without requiring feature scaling or normalization.
   
2. **No Need for Data Scaling**:
   - Unlike other machine learning models such as Support Vector Machines (SVM) or K-Nearest Neighbors (KNN), Random Forest does not require the features to be scaled. This is particularly beneficial for our dataset, as scaling would otherwise add unnecessary complexity to the data preprocessing steps.
   
3. **Robust to Overfitting**:
   - Random Forest is less likely to overfit compared to individual decision trees due to the ensemble learning mechanism, where it creates multiple decision trees and combines their results. This helps in improving the model’s generalization on unseen data.

4. **Interpretability and Feature Importance**:
   - Random Forest provides insight into the importance of each feature in making predictions. This is valuable when trying to understand which features play a significant role in detecting fraudulent transactions.


