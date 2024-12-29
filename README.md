# CreditCard_Fraud_Detection
FraudGuard is a machine learning model for detecting fraudulent credit card transactions using anonymized transaction data.

## Dataset
The dataset used in this project contains anonymized transaction details with a target variable (`1` for fraud, `0` for non-fraud).

### Key Insights:
- **Class Distribution**: The dataset is heavily imbalanced, with a majority of transactions being non-fraudulent.
- **Null Values**: Checked and handled during preprocessing.



### 1. Preprocessing
The preprocessing steps include:
- Checking for missing values


```python
# Import necessary libraries
import pandas as pd



# Check for null values
print(data.isnull().sum())
```

