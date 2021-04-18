import pigpio
from time import sleep

# connect to the 
pi = pigpio.pi()

# loop forever
# 1200, 2000 range up and down
deg = 500
pi.set_servo_pulsewidth(13, 2500)


#while deg <= 2500:
    #pi.set_servo_pulsewidth(13, deg)    # off
    #sleep(0.1)
    #deg += 10

    


    
