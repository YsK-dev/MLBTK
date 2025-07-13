# %%

# grid search vs random search

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

# Define the model
model = RandomForestClassifier(random_state=42) 

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Perform Grid Search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)   

# Get the best parameters and score from Grid Search
best_params_grid = grid_search.best_params_
best_score_grid = grid_search.best_score_   
print("Grid Search Best Parameters:", best_params_grid)
print("Grid Search Best Score:", best_score_grid)

# Perform Randomized Search
random_search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train) 

# Get the best parameters and score from Randomized Search
best_params_random = random_search.best_params_
best_score_random = random_search.best_score_
print("Random Search Best Parameters:", best_params_random) 

print("Random Search Best Score:", best_score_random)   


# %%
# add desicion tree classifier and compare results
from sklearn.tree import DecisionTreeClassifier

# Define the model
dt_model = DecisionTreeClassifier(random_state=42)
# Define the parameter grid for GridSearchCV
dt_param_grid = {
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}       

# Perform Grid Search for Decision Tree
dt_grid_search = GridSearchCV(dt_model, dt_param_grid, cv=5, scoring='accuracy')
dt_grid_search.fit(X_train, y_train)
# Get the best parameters and score from Grid Search for Decision Tree
dt_best_params_grid = dt_grid_search.best_params_
dt_best_score_grid = dt_grid_search.best_score_
print("Decision Tree Grid Search Best Parameters:", dt_best_params_grid)
print("Decision Tree Grid Search Best Score:", dt_best_score_grid)

# Perform Randomized Search for Decision Tree
dt_random_search = RandomizedSearchCV(dt_model, dt_param_grid, n_iter=10, cv=5, scoring='accuracy', random_state=42)
dt_random_search.fit(X_train, y_train)
# Get the best parameters and score from Randomized Search for Decision Tree

dt_best_params_random = dt_random_search.best_params_
dt_best_score_random = dt_random_search.best_score_
print("Decision Tree Random Search Best Parameters:", dt_best_params_random)
print("Decision Tree Random Search Best Score:", dt_best_score_random)


# Evaluate the best models on the test set
best_rf_model = grid_search.best_estimator_
rf_predictions = best_rf_model.predict(X_test)      
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Test Accuracy:", rf_accuracy)      

best_dt_model = dt_grid_search.best_estimator_
dt_predictions = best_dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("Decision Tree Test Accuracy:", dt_accuracy)

# %%    
# svm
from sklearn.svm import SVC
# Define the model
svm_model = SVC(random_state=42)
# Define the parameter grid for GridSearchCV
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}       

# Perform Grid Search for SVM
svm_grid_search = GridSearchCV(svm_model, svm_param_grid, cv=5, scoring='accuracy')
svm_grid_search.fit(X_train, y_train)       

# Get the best parameters and score from Grid Search for SVM
svm_best_params_grid = svm_grid_search.best_params_
svm_best_score_grid = svm_grid_search.best_score_
print("SVM Grid Search Best Parameters:", svm_best_params_grid)
print("SVM Grid Search Best Score:", svm_best_score_grid)       

# Perform Randomized Search for SVM
svm_random_search = RandomizedSearchCV(svm_model, svm_param_grid, n_iter=10, cv=5, scoring='accuracy', random_state=42)
svm_random_search.fit(X_train, y_train)

# Get the best parameters and score from Randomized Search for SVM
svm_best_params_random = svm_random_search.best_params_ 

svm_best_score_random = svm_random_search.best_score_
print("SVM Random Search Best Parameters:", svm_best_params_random)
print("SVM Random Search Best Score:", svm_best_score_random)



# %%
# k-fold and leave one out cross validation
from sklearn.model_selection import cross_val_score, LeaveOneOut        
# Define the model
model = RandomForestClassifier(random_state=42)
# Perform k-fold cross-validation
k_fold_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("K-Fold Cross-Validation Scores:", k_fold_scores)
print("Mean K-Fold Cross-Validation Score:", np.mean(k_fold_scores))    

# Perform Leave-One-Out Cross-Validation
loo = LeaveOneOut()
loo_scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
print("Leave-One-Out Cross-Validation Scores:", loo_scores)
print("Mean Leave-One-Out Cross-Validation Score:", np.mean(loo_scores))
# %%