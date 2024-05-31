import os
import pickle

import txtai
from langchain_community.document_loaders import PyPDFLoader
from tkinter import filedialog

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# This uses langchain PyPDFLoader
# You can also use the Save embeddings and Load embeddings
# The only difference is the pdf loader

class LoadPDF:

    def __init__(self, path="sentence-transformers/nli-mpnet-base-v2"):
        self.embeddings = txtai.Embeddings(path=path)

    def Loader(self):
        loader = PyPDFLoader(self.pdf)
        pages = loader.load_and_split()
        return pages

    def CreateEmbeddings(self):
        self.pdf = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf"), ("PDF Files", "*.PDF")])
        print(self.pdf)
        Content = []
        pages = self.Loader()
        for page in pages:
            Content.append("Source: " + str(page.metadata.get('source')) + "Page :" + str(
                page.metadata.get('page')) + page.page_content)
        name = input("Save embeddings with name...\n")
        print("Please wait while we are processing your file....")
        SavePath = filedialog.askdirectory()
        print(SavePath)
        self.embeddings.index(Content)
        self.embeddings.save(os.path.join(SavePath, name))
        with open(os.path.join(SavePath, name + ".pickle"), "wb") as f:
            pickle.dump(Content, f)

    def Search(self, Query: str, Content):
        idx, confidence = self.embeddings.search(Query, limit=1)[0]
        print(idx)
        print(confidence)
        return Content[int(idx)]

    def SearchEmbeddings(self, Query: str):
        Content = []
        SavePath = filedialog.askdirectory()
        filename = input("Load embeddings with name...\n")
        with open(os.path.join(SavePath, filename) + ".pickle", "rb") as f:
            Content = pickle.load(f)
        self.embeddings.load(os.path.join(SavePath, filename))
        result = self.Search(Query=Query, Content=Content)
        return result
