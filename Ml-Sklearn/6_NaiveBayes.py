#%%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a DataFrame for better visualization
iris_df = pd.DataFrame(data=X, columns=iris.feature_names)
iris_df['target'] = y
print("Iris Dataset:")
print(iris_df.head())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.3, random_state=47, stratify=y # Ensure stratified split for balanced classes
)
# Create a Gaussian Naive Bayes model
gnb_model = GaussianNB()

# Fit the model to the training data
gnb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gnb_model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# %%
