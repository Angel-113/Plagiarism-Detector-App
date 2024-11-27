from functorch.dim import Tensor
from sentence_transformers import SentenceTransformer, util
import numpy as np
from src.FileReader import read_file

# paths for testing
path1_t = "../file_tests/Dynamic Storage Allocation.pdf"
path2_t = "../file_tests/Next Generation Malloc.pdf"

class Detector:

    def __init__(self, path1: str = path1_t, path2: str = path1_t, model: str = 'all-MiniLM-L6-v2'):
        self.__model = SentenceTransformer(model)
        self.__embeddings1 = None
        self.__embeddings2 = None
        self.__set_path(path1, 1)
        self.__set_path(path2, 2)

    def __encode(self, src: str, file: bool = True) -> Tensor:
        try:
            txt = read_file(src) if file else src
            return self.__model.encode(txt, convert_to_tensor=True)
        except Exception as e:
            print(f"Exception : { e }")

    def __large_encode(self, path: str, chunk_size: int = 10) -> list[Tensor]:
        try:
            embeddings: list[Tensor] = []
            txt = read_file(path).splitlines()
            size: int = len(txt)

            if size < chunk_size:
                print("file is to small")
                return []

            chunks: list[str] = []
            for i in range(0, size, chunk_size):
                chunk = ""
                for j in range(i, min(i + chunk_size, size)):
                    chunk += txt[j]
                chunks.append(chunk)
            print("Chunks have been generated ...")

            for chunk in chunks:
                embeddings.append(self.__encode(chunk, file=False))
            print("Embeddings have been generated ...")

            return embeddings
        except Exception as e:
            print(f"Exception : { e }")
            raise e

    def __set_path(self, path: str, n: int) -> None:
        try:
            if n % 2 != 0:
                self.__path1 = path
            else:
                self.__path2 = path
        except path is None:
            print("Can't set a path to a None value")

    def get_embeddings(self) -> (Tensor | list[Tensor], Tensor | list[Tensor]):
        return self.__embeddings1, self.__embeddings2

    def compare_embeddings(self) -> float:
        try:
            size1: int = len(self.__embeddings1)
            size2: int = len(self.__embeddings2)
            if size1 == size2 and size1 == 1:
                return util.cos_sim(self.__embeddings1, self.__embeddings2).item()
            else:
                similarities: list[float] = []
                min_embeddings: list[Tensor] = self.__embeddings2 if size1 > size2 else self.__embeddings1
                max_embeddings: list[Tensor] = self.__embeddings1 if size1 > size2 else self.__embeddings2

                print("Calculating similarity ...")
                for i in range(0, len(max_embeddings) - len(min_embeddings) + 1, len(min_embeddings)):
                    aux = max_embeddings[i:i + len(min_embeddings)]
                    for j in range(0, min(len(aux), len(min_embeddings))):
                        similarities.append(util.cos_sim(aux[j], min_embeddings[j]).item())

                return float(np.mean(similarities))
        except Exception as e:
            print(f"Exception : { e }")
            raise e

    def encode_files(self, src1: str = None, src2: str = None, large: bool = False) -> None:
        try:
            if src1 and src2:
                self.__embeddings1 = self.__encode(src1, file=False)
                self.__embeddings2 = self.__encode(src2, file=False)
            elif large:
                self.__embeddings1 = self.__large_encode(path=self.__path1)
                self.__embeddings2 = self.__large_encode(path=self.__path2)
            else:
                self.__embeddings1 = self.__encode(self.__path1)
                self.__embeddings2 = self.__encode(self.__path2)
        except self.__path1 is None or self.__path2 is None:
            print("paths cannot be None")

    def get_paths(self) -> (str, str):
        return self.__path1, self.__path2

    def change_path(self, path: str, n: int = 1) -> None:
        try:
            if n % 2 > 0:
                self.__set_path(path, n)
            elif n % 2 == 0:
                self.__set_path(path, n)
        except path is None:
            print("Can't change a path to a None value")