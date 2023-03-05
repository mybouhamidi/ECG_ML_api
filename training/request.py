import requests
import pandas as pd
from time import time

import json

test_df = pd.read_csv("demo2.csv", delimiter=";", header=None)
features_df = test_df.astype("float32").transpose()

class_names = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}

# url = "https://ecg-ml-api.onrender.com/predict"
url = "http://localhost:8000/predict"

# print(features_df.to_dict(orient="list"))
previous = time()
y = requests.post(url, json={"data": features_df.to_dict(orient="list")[0]})

w = json.loads(y.text)
print(w["result"], time() - previous)
