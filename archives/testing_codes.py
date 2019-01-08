from datetime import datetime
from threading import Timer
import sys
import time

x=datetime.today()
y=x.replace(day=x.day, hour=18, minute=30, second=0, microsecond=0)
delta_t=y-x

secs=delta_t.seconds+1

def hello_world():
    print("hello world")
    #...

t = Timer(secs, hello_world)
t.start()
for i in range(secs,0,-1):
    print("Seconds left before it begins", i)
    sys.stdout.flush()
    time.sleep(1)

