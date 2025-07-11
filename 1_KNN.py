# %%
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt 


def load_and_prepare_data():
    """Load breast cancer dataset and prepare features and target."""
    cancer = load_breast_cancer()
    
    df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target
    
    x = df[cancer.feature_names]
    y = df['target']
    
    return x, y


def train_knn_model(x_train, y_train, n_neighbors=5):
    """Train KNN classifier with specified number of neighbors."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    return knn


def evaluate_model(model, x_test, y_test):
    """Evaluate the trained model and print results."""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return y_pred


def main():
    """Main function to execute the KNN workflow."""
    from sklearn.model_selection import train_test_split
    # Load and prepare data
    x, y = load_and_prepare_data()
    
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=40
    )
    
    # Train the model
    knn_model = train_knn_model(x_train, y_train, n_neighbors=5)
    
    # Evaluate the model
    predictions = evaluate_model(knn_model, x_test, y_test)
    conf_matrix = confusion_matrix(    y_test, predictions)
    print("\nConfusion Matrix:")
    print(conf_matrix)  

    plt.figure(figsize=(10, 6))
    neighbors_range = range(1, 21)
    accuracy_scores = [train_knn_model(x_train, y_train, n).score(x_test, y_test) for n in neighbors_range]
    plt.plot(neighbors_range, accuracy_scores)
    plt.title('KNN Accuracy vs Number of Neighbors')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, 21))
    plt.grid()
    plt.show()  

# knn  regression
    # Example of KNN regression (not part of the original code)
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.datasets import make_regression
    import numpy as np
    X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
    y = 2 * X.squeeze() + np.random.randn(100) * 2
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_regression    

    X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn_regressor = KNeighborsRegressor(n_neighbors=5)
    knn_regressor.fit(X_train, y_train)
    y_pred = knn_regressor.predict(X_test)

    print(f"Regression Predictions: {y_pred[:5]}")



if __name__ == "__main__":
    main()

# %%
