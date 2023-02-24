from fastapi import FastAPI
from typing import Dict

# import pandas as pd
import numpy as np
import torch
import torch.nn as nn

app = FastAPI()

from pydantic import BaseModel


class Item(BaseModel):
    data: list[float] = None


def nin_block(in_channels, out_channels, kernel_size, padding, strides):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, strides, padding),
        nn.BatchNorm1d(out_channels),
        nn.GELU(),
        nn.Conv1d(out_channels, out_channels, kernel_size=1),
        nn.GELU(),
        nn.Conv1d(out_channels, out_channels, kernel_size=1),
        nn.GELU(),
    )


def get_model():
    return nn.Sequential(  # input shape: (batch_size, 1, 187)
        nin_block(
            1, 48, kernel_size=11, strides=4, padding=0
        ),  # output shape: (batch_size, 48, 44)
        nn.MaxPool1d(3, stride=2),  # output shape: (batch_size, 48, 21)
        nin_block(
            48, 128, kernel_size=5, strides=1, padding=2
        ),  # output shape: (batch_size, 128, 21)
        nn.MaxPool1d(3, stride=2),  # output shape: (batch_size, 128, 10)
        nin_block(
            128, 256, kernel_size=3, strides=1, padding=1
        ),  # output shape: (batch_size, 256, 10)
        nn.MaxPool1d(3, stride=2),  # output shape: (batch_size, 256, 4)
        nn.Dropout(0.4),
        # last layers for classification of 5 classes
        nin_block(
            256, 5, kernel_size=3, strides=1, padding=1
        ),  # output shape: (batch_size, 5, 4)
        nn.AdaptiveAvgPool1d(1),  # output shape: (batch_size, 5, 1)
        nn.Flatten(),  # output shape: (batch_size, 5)
    )


@app.post("/predict")
async def classify(heartbeat: Item):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device=device)
    model.load_state_dict(torch.load("best_model0.99.pt"))

    class_names = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
    item = heartbeat.dict()
    # x = heartbeat['data']
    x = np.array(item["data"], dtype="f").reshape(1, 1, 187)

    y = model(torch.tensor(x).to(device=device))
    y = torch.argmax(y, dim=1).cpu().numpy()

    i = y.tolist()
    return {"result": class_names[i[0]]}

