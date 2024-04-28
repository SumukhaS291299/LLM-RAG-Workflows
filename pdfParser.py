import os
from tkinter import filedialog
from RAGLogger import Log
from tika import parser
from fileFilter import PDF


def contentParser(CONTENT):
    lines = CONTENT.split("\n")
    out = []
    for line in lines:
        if line == '\n' or line == '':
            pass
        else:
            out.append(line)
    return out


def IndexLines():
    logger = Log("Parser").log()
    dir = filedialog.askdirectory(title="Directory containing PDFs")
    files = os.listdir(dir)
    filteredFiles = filter(PDF, files)
    OUT = []
    for file in filteredFiles:
        logger.info("Parced file: " + str(file))
        parsed_document = parser.from_file(os.path.join(dir, file))
        OUT = contentParser(parsed_document.get("content"))
        # pprint(parsed_document.get("metadata"))
        # pprint(OUT)
    return OUT
