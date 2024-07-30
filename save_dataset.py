# save_dataset.py

import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Save dataset to CSV
data.to_csv('iris.csv', index=False)
