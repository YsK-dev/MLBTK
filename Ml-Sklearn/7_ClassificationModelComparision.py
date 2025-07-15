# %%

from sklearn.datasets import make_classification, make_moons, make_circles
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Create a synthetic dataset for classification

X, y = make_classification(
    n_samples=100,
    n_features=10,
    n_informative=2,
    n_redundant=1,
    n_clusters_per_class=1,
    random_state=42
)

Xy = (X, y)  # Store the dataset in a tuple for later use
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=20)
#plt.title('Synthetic Classification Dataset')
#plt.xlabel('Feature 1')
#plt.ylabel('Feature 2')
#plt.show()

dataset = [Xy,
           make_moons(n_samples=100, noise=0.2, random_state=42),
           make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)]

fig = plt.figure(figsize=(12, 4))
for i, (X, y) in enumerate(dataset):
    ax = fig.add_subplot(1, 3, i + 1)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=20)
    ax.set_title(f'Dataset {i + 1}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2') 

names = ["Nearest Neighbors",
         "Linear SVM",
         "RBF SVM",
         "Decision Tree",
         "Random Forest",
         "Naive Bayes"]

classifiers = [KNeighborsClassifier(n_neighbors=3),
               SVC(kernel="linear", C=0.025),
               SVC(gamma=2, C=1),
               DecisionTreeClassifier(max_depth=5),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=10),
               GaussianNB()]  

fig = plt.figure(figsize=(9, 12))
for ds_count, (X, y) in enumerate(dataset):
    ax = fig.add_subplot(3, 3, ds_count + 1)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=20)
    ax.set_title(f'Dataset {ds_count + 1}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

    for name, clf in zip(names, classifiers):
        clf.fit(X, y)
        score = clf.score(X, y)
        print(f"{name} accuracy on Dataset {ds_count + 1}: {score:.2f}")
        ax.text(0.5, 0.1 + 0.1 * classifiers.index(clf), 
                f"{name}: {score:.2f}", 
                transform=ax.transAxes, 
                fontsize=8, 
                ha='center', 
                bbox=dict(facecolor='white', alpha=0.5))
plt.tight_layout()
plt.show()

        

# %% [markdown]


# %%
