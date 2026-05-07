import urllib.request
import os

MODEL_URL = "https://github.com/PINTO0309/PINTO_model_zoo/raw/main/083_MiDaS/models/midas_v21_small_256x256.tflite"
SAMPLE_IMG_URL = "https://raw.githubusercontent.com/intel-isl/MiDaS/master/input/dog.jpg"

def download_file(url, filename):
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping.")
        return
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print("Done.")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    download_file(MODEL_URL, "midas_small.tflite")
    download_file(SAMPLE_IMG_URL, "sample.jpg")
