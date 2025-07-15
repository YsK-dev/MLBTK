# %%
from sklearn.datasets import load_diabetes

from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV

# Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target 

# ridge regression
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
ridge_grid_search = GridSearchCV(ridge, ridge_param_grid, cv=5)
ridge_grid_search.fit(X_train, y_train)
print(f"Best Ridge Parameters: {ridge_grid_search.best_params_}")  
print(f"Best Ridge Score: {ridge_grid_search.best_score_:.4f}") 
ridge_score = ridge.score(X_test, y_test)
print(f"Ridge Regression Score: {ridge_score:.4f}") 

# lasso regression
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
lasso_score = lasso.score(X_test, y_test)       

print(f"Lasso Regression Score: {lasso_score:.4f}") 

# %%
# elastic net
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)
elastic_net_score = elastic_net.score(X_test, y_test)
print(f"Elastic Net Score: {elastic_net_score:.4f}")        

# 

# %%
