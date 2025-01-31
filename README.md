
# Kaggle Apple 2024 Sales Dataset

- [Apple 2024 Sales Dataset](https://www.kaggle.com/datasets/chozenwon/apple-2024-sales-dataset)

# Import Libaries and Load Dataset

```python
# Import required libraries first
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings 
warnings.filterwarnings("ignore")

# Then import kaggle related libraries
from dotenv import load_dotenv
load_dotenv()

import kaggle
kaggle.api.authenticate()

import kagglehub
import os

# Now load the data
path = kagglehub.dataset_download("ayushcx/apple-sales-dataset-2024")

# Find the CSV file in the directory
csv_path = os.path.join(path, "apple_sales_2024.csv")
df = pd.read_csv(csv_path)
print(df.head())
```
![Dataset Overview](images/dataset_overview.png)


# Data Cleaning and Preprocessing

```python
# Check for missing values
df.isnull().sum()

```
State                              0
Region                             0
iPhone Sales (in million units)    0
iPad Sales (in million units)      0
Mac Sales (in million units)       0
Wearables (in million units)       0
Services Revenue (in billion $)    0
dtype: int64

Dataset returns clean with no missing values

# Exploratory Data Analysis

We will explore the dataset using a couple visualizations options to uncover insights and patterns in data

## Visualizations


### 1. Correlation Heatmap
![Correlation Heatmap](images/correlation_heatmap.png)

A correlation heatmap visualizes the relationships between numerical variables in your dataset, providing several key insights:
1. Correlation strength: The color intensity represents the strength of the correlation between two variables. Strong correlations are represented by darker shades, while weaker correlations are lighter. Values range from -1 to +1.

- +1 Dark Red: Strong positive correlation
- -1 Dark Blue: Strong negative correlation
- 0 Light Gray: No correlation

### 2. Pair Plot Analysis
![Pair Plot](images/pairplot.png)

### 3. Regional iPhone Sales
![iPhone Sales by Region](images/iphone_sales_region.png)

# Predictive Modeling

We can try to predict one of the sales figures compared to the other. We will use a linear regression model to predict iPhone Sales 

```python
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define features and target variable
X = df[['iPad Sales (in million units)', 'Mac Sales (in million units)', 'Wearables (in million units)', 'Services Revenue (in billion $)']]
y = df['iPhone Sales (in million units)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse, r2

```
## Prediction Result Explanation

**Result: (53.05679117142261, -0.013370467073819947)**

**MSE (Mean Squared Error) = 53.05679**

This measures the average squared difference between predicted and actual iPhone sales
The units are squared (million units²), so this is a large error in prediction
Root MSE would be about 7.28 million units of error on average

**R² Score = -0.01337**
R² normally ranges from 0 to 1 where 1 is a perfect fit and 0 is no fit at all. Negative values occur when model performs worse than horizontal line (no fit)

The score incdicates that the model is performing poorly and it would be better to use the mean of iPhone sales as a prediction. 

## Conclusion

There might not be a strong linear relationship between iPhone sales and other products. 

## Future Work and Improvements

1. Consider non-linear models
2. Add more features to the model
3. Try other algorithms like Random Forest or Gradient Boosting