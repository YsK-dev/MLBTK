#%%
import pandas as pd 
from ucimlrepo import fetch_ucirepo
import numpy as np

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 

# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 

# Create combined dataframe to handle missing values consistently
data_combined = pd.concat([X, y], axis=1)
data_cleaned = data_combined.dropna()

# Split back into features and target
X_clean = data_cleaned.drop('num', axis=1)
y_clean = data_cleaned['num']

print(f"Cleaned data shapes - X: {X_clean.shape}, y: {y_clean.shape}")

#%%
# split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
)

#%%
from sklearn.linear_model import LogisticRegression

# create a logistic regression model
logistic_model = LogisticRegression(max_iter=1000, random_state=42)

# fit the model to the training data
logistic_model.fit(X_train, y_train)

# make predictions on the test set
y_pred = logistic_model.predict(X_test)

#%%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': logistic_model.coef_[0]
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)
# Plot feature importances
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))

plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances from Logistic Regression')
plt.show()  


# %%
