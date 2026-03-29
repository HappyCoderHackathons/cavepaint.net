import lgpio
import time

chip = lgpio.gpiochip_open(1)

IN1 = 98
IN2 = 91
C1  = 92

PULSES_PER_REV = 661
OVERSHOOT_COMPENSATION = 330  # increase this if still overshooting

lgpio.gpio_claim_output(chip, IN1)
lgpio.gpio_claim_output(chip, IN2)
lgpio.gpio_claim_input(chip, C1)

def forward():
    lgpio.gpio_write(chip, IN1, 1)
    lgpio.gpio_write(chip, IN2, 0)

def backward():
    lgpio.gpio_write(chip, IN1, 0)
    lgpio.gpio_write(chip, IN2, 1)

def stop():
    lgpio.gpio_write(chip, IN1, 0)
    lgpio.gpio_write(chip, IN2, 0)

def rotate_degrees(degrees):
    target_pulses = int((abs(degrees) / 360) * PULSES_PER_REV)
    target_pulses = max(0, target_pulses - OVERSHOOT_COMPENSATION)
    pulse_count = 0
    last_state = lgpio.gpio_read(chip, C1)

    print(f"Rotating {degrees} degrees ({target_pulses} pulses)...")

    if degrees > 0:
        forward()
    else:
        backward()

    while pulse_count < target_pulses:
        current = lgpio.gpio_read(chip, C1)
        if current == 1 and last_state == 0:
            pulse_count += 1
        last_state = current

    stop()
    print(f"Done. Pulses counted: {pulse_count}")

rotate_degrees(360)
lgpio.gpiochip_close(chip)
