import json
from fastapi import FastAPI
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from pyod.models.knn import KNN
import os

app = FastAPI()

neigh = None
clf = None

@app.on_event("startup")
def load_train_model():
    df = pd.read_csv("iris_ok.csv")
    global neigh
    neigh = KNeighborsClassifier(n_neighbors=len(np.unique(df["y"])))
    neigh.fit(df[df.columns[:4]].values.tolist(), df["y"])
    print("Model finished training.")

    global clf
    clf_name = "KNN"
    clf = KNN()
    clf.fit(df[df.columns[:4]].values.tolist(), df["y"])

@app.get("/predict")
def predict(p1: float, p2: float, p3: float, p4: float):
    pred = neigh.predict([[p1,p2,p3,p4]])[0]
    return "{}".format(pred)

@app.get("/detect")
def detect(p1: float, p2: float, p3: float, p4: float):
    detect = clf.predict([[p1,p2,p3,p4]],return_confidence=True)
    return json.dumps({"decision": detect[0].tolist(), "accuracy": detect[1].tolist()})

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ['HOST'], port = os.environ['PORT'])