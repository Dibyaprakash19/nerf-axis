import sys
import os
import cv2

# Add current dir to path to import lightweight_scanner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lightweight_scanner import MiDaSScanner

def test_bg():
    scanner = MiDaSScanner("midas_small.tflite")
    
    input_img = "sample.jpg"
    output_img = "sample_clean_hybrid.png"
    
    if not os.path.exists(input_img):
        print(f"Error: {input_img} not found.")
        return

    print(f"Processing {input_img}...")
    scanner.remove_background_hybrid(input_img, output_img)
    print(f"Done! Result saved to {output_img}")

if __name__ == "__main__":
    test_bg()
