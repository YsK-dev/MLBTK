from sklearn.datasets import load_breast_cancer
import pandas as pd


cancer = load_breast_cancer()


df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)

df['target'] = cancer.target
