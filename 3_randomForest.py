# %%
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import matplotlib.pyplot as plt 

oli = fetch_olivetti_faces()

plt.figure(figsize=(10, 6))
for i in range(10): 
    plt.subplot(2, 5, i + 1)
    plt.imshow(oli.images[i], cmap='gray')
    plt.title(f"Image {i + 1}")
    plt.axis('off')
plt.tight_layout()
plt.show()

x = oli.data
y = oli.target 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier 

rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
rf_model.fit(x_train, y_train)

y_pred = rf_model.predict(x_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)       
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred)) 

# make n_estimators random and find best n_estimators number
n_estimators_range = range(1, 21)
accuracies = []
for n in n_estimators_range:
    rf_model = RandomForestClassifier(n_estimators=n, random_state=42)
    rf_model.fit(x_train, y_train)
    y_pred = rf_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy) 

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, accuracies, marker='o')
plt.title('Random Forest Accuracy vs Number of Estimators')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.xticks(n_estimators_range)
plt.grid()
plt.show()  
# %%
