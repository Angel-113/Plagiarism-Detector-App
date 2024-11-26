from src.Detector import Detector

def main() -> None:
    detector = Detector(
        "/home/angel/Downloads/Next Generation Malloc.pdf",
        "/home/angel/Downloads/Dynamic Storage Allocation.pdf"
    )
    detector.encode_files()
    print(detector.compare_embeddings())

if __name__ == '__main__':
    main()