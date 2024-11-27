from typing import Any
from src.Detector import Detector
import nicegui

from src.FileReader import read_file

def clear_set_log( log: nicegui.ui.log, data: Any) -> None:
    log.clear()
    log.push(Any)

class App:
    def __init__(self):
        self.detector = Detector()
        self.gui = nicegui.ui

    def __document_similarity(self, path1: str, path2: str) -> float:
        self.detector.change_path(path1, n=1)
        self.detector.change_path(path2, n=2)
        self.detector.encode_files(large=True)
        return self.detector.compare_embeddings()

    def __start(self) -> None:

        with self.gui.tabs().classes('w-full') as tabs:

            one = self.gui.tab('Document Similarity')

        with self.gui.tab_panels(tabs, value=one).classes('w-full'):

            with self.gui.tab_panel(one):

                self.gui.label("Enter file paths to compare their similarity.").classes("text-2xl font-bold")

                with self.gui.row():

                    input1 = self.gui.textarea("file 1", placeholder="Enter path to file here...")
                    input2 = self.gui.textarea("file 2", placeholder="Enter path to file here...")
                    log = self.gui.log(max_lines=10).classes('w-full h-20')
                    self.gui.button("Compare", on_click=lambda: clear_set_log(log, self.__document_similarity(str(input1.value), str(input2.value))))

        return

    def run(self) -> None:
        self.__start()
        self.gui.run()
        return