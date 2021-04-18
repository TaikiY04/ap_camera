import numpy as np
import pigpio
import time
from time import sleep
from math import sin, cos, radians

pi = pigpio.pi()
pi.set_servo_pulsewidth(13, 1500)
pi.set_servo_pulsewidth(26, 1700)
