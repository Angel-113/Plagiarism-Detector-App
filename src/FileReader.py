from PyPDF2 import PdfReader
import os

def read_pdf(path: str) -> str:
    try:
        text = ""
        reader = PdfReader(path)
        for page in reader.pages:
            text += page.extract_text()
        return text
    except FileNotFoundError or path is None:
        print("Can't open the file")

def read_txt(path: str) -> str:
    try:
        text: str = ""
        file = open(path, "r")
        lines: list[str] = file.readlines()
        for line in lines:
            text += line
        file.close()
        return text
    except FileNotFoundError or path is None:
        print("Can't open the file")

def read_file(path: str) -> str:
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    elif ext == ".txt":
        return read_txt(path)
    else:
        print("Extension not supported")
        return ""