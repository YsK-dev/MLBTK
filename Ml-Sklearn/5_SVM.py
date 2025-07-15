#%%
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


digits = load_digits()

import matplotlib.pyplot as plt

fig, axes =plt.subplots(nrows=2, ncols=5, figsize=(10, 5),subplot_kw={'xticks':[], 'yticks':[]})

for ax, image, label in zip(axes.ravel(), digits.images, digits.target):
    ax.imshow(image, cmap='gray', interpolation='nearest')
    ax.set_title(label) 

plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42,
    stratify=digits.target
)
# Create a Support Vector Machine model
svm_model = SVC(kernel='linear', random_state=47)


# Fit the model to the training data
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# %%
