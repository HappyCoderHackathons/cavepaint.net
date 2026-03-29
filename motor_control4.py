import lgpio
import time

chip = lgpio.gpiochip_open(1)

IN1 = 98
IN2 = 91
C1  = 92

lgpio.gpio_claim_output(chip, IN1)
lgpio.gpio_claim_output(chip, IN2)
lgpio.gpio_claim_input(chip, C1)

def forward():
    lgpio.gpio_write(chip, IN1, 1)
    lgpio.gpio_write(chip, IN2, 0)

def stop():
    lgpio.gpio_write(chip, IN1, 0)
    lgpio.gpio_write(chip, IN2, 0)

pulse_count = 0
last_state = lgpio.gpio_read(chip, C1)

print("Motor starting...")
forward()

start = time.time()
while time.time() - start < 5:
    current = lgpio.gpio_read(chip, C1)
    if current == 1 and last_state == 0:
        pulse_count += 1
    last_state = current

stop()

print(f"Motor stopped")
print(f"Pulses counted: {pulse_count}")

lgpio.gpiochip_close(chip)
