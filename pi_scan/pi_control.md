# Pi 3D Scanner: Remote Control & Deployment

This guide covers how to sync your code to the Raspberry Pi and control the stepper motor remotely.

## 1. Remote Setup
Ensure your Raspberry Pi is connected to the same network and accessible via `gp5.local`.

## 2. Sync Code to Pi (SCP)
Run this command from your local Mac terminal to sync the `pi_scan` directory:

```bash
# Sync local pi_scan to the Pi home directory
scp -r ./pi_scan spass@gp5.local:~/nerf-axis/
```

## 3. Remote Execution (SSH)
To run the Flask app on the Pi:

```bash
ssh spass@gp5.local "cd ~/nerf-axis/pi_scan && source venv/bin/activate && python3 app.py"
```

## 4. Stepper Motor Rotation Commands
To rotate the turntable 90 degrees without opening the web UI:

```bash
# Method A: Run the standalone script
ssh spass@gp5.local "cd ~/nerf-axis/pi_scan && source venv/bin/activate && python3 stepper90.py"

# Method B: Trigger via API (if app is running)
curl -X POST http://gp5.local:5000/rotate -H "Content-Type: application/json" -d '{"degrees": 90}'
```

## 5. Wiring Reminder (ULN2003)
| Pi GPIO | ULN2003 |
|---------|---------|
| GPIO17  | IN1     |
| GPIO18  | IN2     |
| GPIO27  | IN3     |
| GPIO22  | IN4     |
| 5V      | VCC     |
| GND     | GND     |
