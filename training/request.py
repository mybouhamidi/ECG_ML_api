import requests
import pandas as pd
from time import time
import numpy as np
import json

test_df = pd.read_csv("ML-data/mitbih_test.csv", header=None)
features_df = test_df.iloc[:, :-1]
features_df = features_df.astype("float32")
X_test_np = features_df.to_numpy()
X_test_np = np.reshape(X_test_np, (X_test_np.shape[0], 1, X_test_np.shape[1]))
X_test_np = (X_test_np - X_test_np.mean()) / X_test_np.std()

class_names = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}

x = X_test_np[:1, :, :]
x = x.reshape(1, 1, 187)

w = list(x[0][0].tolist())


# url = "https://ecg-ml-api.onrender.com/predict"
url = "http://localhost:8000/predict"

previous = time()
y = requests.post(url, json={"data": w})
w = json.loads(y.text)
print(w["result"], time() - previous)
