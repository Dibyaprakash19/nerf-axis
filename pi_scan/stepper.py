import time
import argparse
import RPi.GPIO as GPIO

class Stepper:
    """
    Controller for the 28BYJ-48 stepper motor via ULN2003 driver.
    """
    PINS = [17, 18, 27, 22]
    SEQUENCE = [
        [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0],
        [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 1]
    ]

    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        for pin in self.PINS:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, False)

    def rotate(self, degrees=90, step_delay=0.0015):
        """Rotates the motor by the specified degrees."""
        steps = int(round(abs(degrees) / 360.0 * 4096))
        direction = 1 if degrees > 0 else -1
        seq = self.SEQUENCE if direction > 0 else list(reversed(self.SEQUENCE))
        
        try:
            for _ in range(steps):
                for state in seq:
                    for i, pin in enumerate(self.PINS):
                        GPIO.output(pin, state[i])
                    time.sleep(step_delay)
        finally:
            self.release()

    def release(self):
        """Releases all pins to prevent overheating."""
        for pin in self.PINS:
            GPIO.output(pin, False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("degrees", type=int, nargs='?', default=90)
    args = parser.parse_args()
    
    motor = Stepper()
    try:
        print(f"Rotating {args.degrees} degrees...")
        motor.rotate(args.degrees)
        print("Done.")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
