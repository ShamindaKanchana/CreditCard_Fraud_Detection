# CreditCard_Fraud_Detection
FraudGuard is a machine learning model for detecting fraudulent credit card transactions using anonymized transaction data.

## Dataset
The dataset used in this project contains anonymized transaction details with a target variable (`1` for fraud, `0` for non-fraud).

### Key Insights:
- **Class Distribution**: The dataset is heavily imbalanced, with a majority of transactions being non-fraudulent.
- **Null Values**: Checked and handled during preprocessing.



## Key Insights from the Dataset

1. **Transaction Type Distribution**:
   - According to the dataset, **99.4% of transactions are withdrawals**, while the remaining **0.6%** represent other types of transactions.
   - This highlights that the majority of the activity in the dataset involves customers withdrawing money.

2. **Transaction Amount Distribution**:
   - A significant number of transactions have an amount **less than $100**.
   - This indicates that most customers are engaged in smaller transactions, which might impact the patterns for fraudulent vs. non-fraudulent behaviors.


### 1. Preprocessing
The preprocessing steps include:
- Checking for missing values


```python
# Import necessary libraries
import pandas as pd



# Check for null values
print(data.isnull().sum())
```

