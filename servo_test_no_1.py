import RPi.GPIO as GPIO
from gpiozero import Servo
import time
from time import sleep

GPIO.setmode(GPIO.BOARD)

GPIO.setup(37,GPIO.OUT)
pwm = GPIO.PWM(37, 50)
pwm.start(0)

def SetAngle(angle):
    duty = angle / 18 + 2
    GPIO.output(37, True)
    pwm.ChangeDutyCycle(duty)
    sleep(1)
    GPIO.output(37, False)
    pwm.ChangeDutyCycle(0)

deg = 0

while deg <= 180:
    SetAngle(deg)
    deg += 10
    sleep(1)

pwm.stop()
GPIO.cleanup()
print("end")
