from functorch.dim import Tensor
from sentence_transformers import SentenceTransformer, util
from numpy import ndarray
from src.FileReader import read_file

class Detector:

    def __init__(self, path1: str = None, path2: str = None):
        self.__model = SentenceTransformer('all-MiniLM-L6-v2')
        self.__embeddings1 = None
        self.__embeddings2 = None
        self.__set_path(path1, 1)
        self.__set_path(path2, 2)

    def __encode(self, src: str, file: bool = True) -> Tensor | ndarray | list[Tensor]:
        try:
            txt = read_file(src) if file else src
            return self.__model.encode(txt, convert_to_tensor=True)
        except Exception as e:
            print(f"Exception : { e }")
            return []

    def __large_encode(self, path: str) -> Tensor | ndarray | list[Tensor]:
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
                encodings += self.__encode(chunk, file=False)
            return encodings

        except Exception as e:
            print(f"Exception : { e }")
            return []

    def __set_path(self, path: str, n: int) -> None:
        try:
            if n % 2:
                self.__path1 = path
            else:
                self.__path2 = path
        except path is None:
            print("Can't set a path to a None value")

    def compare_embeddings(self) -> float:
        try:
            return util.cos_sim(self.__embeddings1, self.__embeddings2).item()
        except Exception as e:
            print(f"Exception : { e }")
            return 0.0

    def encode_files(self, large: bool = False) -> None:
        try:
            if large:
                self.__embeddings1 = self.__large_encode(path=self.__path1)
                self.__embeddings2 = self.__large_encode(path=self.__path2)
            else:
                self.__embeddings1 = self.__encode(self.__path1)
                self.__embeddings2 = self.__encode(self.__path2)
        except self.__path1 is None or self.__path2 is None:
            print("paths cannot be None")

    def change_path(self, path: str, n: int = 1) -> None:
        try:
            if n % 2 > 0:
                self.__set_path(path, n)
            elif n % 2 == 0:
                self.__set_path(path, n)
        except path is None:
            print("Can't change a path to a None value")