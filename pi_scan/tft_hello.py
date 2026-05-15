from PIL import Image, ImageDraw
from st7735_direct_hello import DirectST7735, WIDTH, HEIGHT

def main():
    display = DirectST7735()
    
    img = Image.new("RGB", (WIDTH, HEIGHT), "red")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, WIDTH-1, HEIGHT-1), outline="yellow")
    draw.text((20, 70), "ST7735 Ready", fill="white")
    
    display.display(img)
    print("Test pattern displayed.")
    display.close()

if __name__ == "__main__":
    main()
