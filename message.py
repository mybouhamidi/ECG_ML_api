from fastapi import FastAPI
from typing import List
import numpy as np

import pandas as pd

from pydantic import BaseModel
from tensorflow import keras

app = FastAPI()

from pydantic import BaseModel


class Item(BaseModel):
    data: List[float]


model = keras.models.load_model("bestmodel")
class_names = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}


@app.post("/predict")
async def classify(heartbeat: Item):
    item = heartbeat.dict()

    predictions = model.predict(pd.DataFrame(item["data"]).transpose())
    predictions = np.argmax(predictions, axis=1)

    i = predictions.tolist()
    return {"result": class_names[i[0]]}
