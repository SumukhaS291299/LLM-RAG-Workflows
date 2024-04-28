import os
import pickle

import torch
from RAGLogger import Log
import txtai
from tkinter import filedialog

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logger = Log("Load_Embeddings").log()

embeddings = txtai.Embeddings(path="sentence-transformers/nli-mpnet-base-v2")
logger.info("Torch CUDA installed: " + str(torch.cuda.is_available()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device selected: " + str(device))

name = filedialog.askdirectory(initialdir=os.path.join("saved", "embeddings"), title="Load your personal data")
embeddings.load(name)
logger.info(f"Loaded embeddings {name}....")

filename = os.path.basename(name)

with open(os.path.join(os.path.join("saved", "data"), filename) + ".pickle", "rb") as f:
    FullData = pickle.load(f)
Run = True

while Run:
    Question = input("Ask any questions related to the document:")
    if Question.strip().lower() == "quit":
        Run = False
    else:
        for idx, confidence in embeddings.search("Basic salary"):
            print(idx)
            print(confidence)
            print(FullData[int(idx)])
