#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# y = mnxn + .... m1x1 + m0x0 + b0
#multivariable linear regression


# make a random dataset
X = np.random.rand(100, 2)  # 100 samples, 3 features
coef = np.array([3,5])  # Coefficients for the features
y = np.dot(X, coef) + np.random.normal(0, 0.1, 100)  # Linear relation with some noise


"""
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='b', label='Data Points')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target Variable')
ax.set_title('3D Scatter Plot of Data Points')
ax.legend()
plt.show()"""

LinearRegressionModel = LinearRegression()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Fit the model to the training data

LinearRegressionModel.fit(X_train, y_train) 
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='b', label='Data Points')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target Variable')

np.meshgrid = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 10),
                     np.linspace(X[:, 1].min(), X[:, 1].max(), 10))
X_grid = np.c_[np.ravel(np.meshgrid[0]), np.ravel(np.meshgrid[1])]
y_grid = LinearRegressionModel.predict(X_grid)
ax.plot_trisurf(X_grid[:, 0], X_grid[:, 1], y_grid, color='r', alpha=0.5, label='Regression Plane')
ax.set_title('3D Scatter Plot with Regression Plane')
ax.legend()
plt.show()
# Make predictions on the test set
y_pred = LinearRegressionModel.predict(X_test)
# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")   



# %%
