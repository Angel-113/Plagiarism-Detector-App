from functorch.dim import Tensor
from sentence_transformers import SentenceTransformer, util
from numpy import ndarray
from FileReader import *

class Detector:

    embeddings1: Tensor | ndarray | list[Tensor] = None
    embeddings2: Tensor | ndarray | list[Tensor] = None
    path1: str = None
    path2: str = None
    model = None

    def __init__(self, path1: str, path2: str):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.__set_path__(path1, 1)
        self.__set_path__(path2, 2)
        return

    def __encode__(self, path: str) -> Tensor | ndarray | list[Tensor]:
        try:
            txt = read_file(path)
            return self.model.encode(txt, convert_to_tensor=True)
        except Exception as e:
            print(f"Exception : {e}")
            return []

    def __large_encode__(self, path: str) -> Tensor | ndarray | list[Tensor]:
        try:
            encodings: list[Tensor] = []
            txt = read_file(path).splitlines()
            chunks: list[str] = []

            size = len(txt)
            for i in range(1, size, 10):
                chunk = ""
                for j in range(i, size, 10):
                    chunk += txt[j]
                chunks += [chunk]

            for chunk in chunks:
                encodings += self.__encode__(chunk)
            return encodings

        except Exception as e:
            print(f"Exception : {e}")
            return []

    def __set_path__(self, path: str, n: int) -> None:
        try:
            if n % 2:
                self.path1 = path
            else:
                self.path2 = path
        except path is None:
            print("Can't set a path to a None value")

    @classmethod
    def compare(cls) -> float:
        try:
            return util.cos_sim(cls.embeddings1, cls.embeddings2).item()
        except cls.embeddings1 is None or cls.embeddings2 is None:
            print("Encodings cannot be None")
            return 0.0

    @classmethod
    def encode_files(cls, large: bool) -> None:
        try:
            if large:
                cls.embeddings1 = cls.__large_encode__(cls.path1)
                cls.embeddings2 = cls.__large_encode__(cls.path2)
            else:
                cls.embeddings1 = cls.__encode__(cls.path1)
                cls.embeddings2 = cls.__encode__(cls.path2)
        except cls.path1 is None or cls.path2 is None:
            print("Encodings cannot be None")