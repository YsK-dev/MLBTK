from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split



iris = load_iris()

X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

# decision tree model train 
from sklearn.tree import DecisionTreeClassifier 

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='gini',max_depth=5, random_state=42)

clf.fit(X_train, y_train) 

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
y_pred = clf.predict(X_test)   
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize the decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree Visualization")
plt.show()

# feature importance
import pandas as pd 
feature_importances = clf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importances:")
print(importance_df)        

for i in range(len(importance_df)):
    print(f"Feature: {importance_df['Feature'].iloc[i]}, Importance: {importance_df['Importance'].iloc[i]:.4f}")
# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances from Decision Tree')
plt.show()

# %%

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd 

import matplotlib.pyplot as plt

load_iris = load_iris()

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    X = load_iris.data[:, pair]
    y = load_iris.target

    # Create a Decision Tree Classifier
    clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    clf.fit(X, y)

    plt.figure(figsize=(8, 6))
    plot_tree(clf, filled=True, feature_names=[load_iris.feature_names[i] for i in pair],
              class_names=load_iris.target_names)
    plt.title(f"Decision Tree Visualization for Features {pair}")
    plt.show()

# %%
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error, r2_score
 
diabets = load_diabetes()
X = diabets.data
y = diabets.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

tree_regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_regressor.fit(X_train, y_train)    

y_pred = tree_regressor.predict(X_test)

mean_squared_error = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error ** 0.5
print(f"Root Mean Squared Error: {rmse:.4f}")   
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mean_squared_error:.4f}")
print(f"R^2 Score: {r2:.4f}")   

# %%