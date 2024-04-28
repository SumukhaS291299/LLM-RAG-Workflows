import os.path
import pickle

import torch
from RAGLogger import Log
import txtai
from pdfParser import IndexLines

logger = Log("Save_Embeddings").log()

embeddings = txtai.Embeddings(path="sentence-transformers/nli-mpnet-base-v2")
logger.info("Torch CUDA installed: " + str(torch.cuda.is_available()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device selected: " + str(device))

logger.info("Generating Index...")
Lines = IndexLines()
embeddings.index(Lines)
logger.info("Saving Index....")
name = input("Save embeddings with name...\n")
logger.info(f"Saving embeddings {name}....")
embeddings.save(os.path.join(os.path.join("saved", "embeddings"), name))
if os.path.exists(os.path.join("saved", "data")):
    pass
else:
    os.mkdir(os.path.join("saved", "data"))
    with open(os.path.join(os.path.join("saved", "data"), name + ".pickle"), "wb") as f:
        pickle.dump(Lines, f)
