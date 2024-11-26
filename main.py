from src.Detector import Detector

def main() -> None:
    detector = Detector()
    detector.encode_files()
    print(detector.compare_embeddings())

if __name__ == '__main__':
    main()