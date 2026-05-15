import time
import signal
import sys
from PIL import Image, ImageDraw
from st7735_direct_hello import DirectST7735, WIDTH, HEIGHT

def main():
    display = DirectST7735()
    signal.signal(signal.SIGINT, lambda s, f: (display.close(), sys.exit(0)))
    
    x, y = WIDTH // 2, HEIGHT // 2
    dx, dy = 4, 3
    radius = 12
    
    while True:
        img = Image.new("RGB", (WIDTH, HEIGHT), "black")
        draw = ImageDraw.Draw(img)
        
        x += dx
        y += dy
        
        if x - radius < 0 or x + radius > WIDTH:
            dx = -dx
        if y - radius < 0 or y + radius > HEIGHT:
            dy = -dy
            
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="blue", outline="white")
        display.display(img)
        time.sleep(0.015)

if __name__ == "__main__":
    main()
