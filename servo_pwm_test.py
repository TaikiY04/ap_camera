import pigpio
import time

pi = pigpio.pi()
pi.set_PWM_dutycycle(13, 0)
pi.set_servo_pulsewidth(13, 500)
