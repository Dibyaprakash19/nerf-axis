import RPi.GPIO as GPIO
import time
import sys

# GPIO pins connected to ULN2003 IN1-IN4
IN1 = 17
IN2 = 18
IN3 = 27
IN4 = 22

pins = [IN1, IN2, IN3, IN4]

def rotate_90(direction=1):
    GPIO.setmode(GPIO.BCM)
    for pin in pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, False)

    # Half-step sequence
    step_sequence = [
        [1,0,0,0],
        [1,1,0,0],
        [0,1,0,0],
        [0,1,1,0],
        [0,0,1,0],
        [0,0,1,1],
        [0,0,0,1],
        [1,0,0,1]
    ]
    
    if direction == -1:
        step_sequence = list(reversed(step_sequence))

    # 28BYJ-48: ~4096 half-steps per full rotation
    steps_for_90_deg = 128

    try:
        for i in range(steps_for_90_deg):
            for step in step_sequence:
                for pin in range(4):
                    GPIO.output(pins[pin], step[pin])
                time.sleep(0.001)
    finally:
        # Relax the motor pins to prevent overheating
        for pin in pins:
            GPIO.output(pin, False)
        # Note: We don't call GPIO.cleanup() here if used as a module
        # but for a standalone script it's fine.

if __name__ == "__main__":
    try:
        rotate_90()
        print("Rotated 90 degrees.")
    finally:
        GPIO.cleanup()
