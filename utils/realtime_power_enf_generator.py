#Using Audacity to collect hourly ENF values

import sounddevice as sd
#import librosa
import time
import threading
import soundfile as sf
from datetime import datetime
#from threading import Timer


duration = 10
fs = 8000

x=datetime.today()
y=x.replace(day=x.day, hour=17, minute=21, second=0, microsecond=0)
delta_t=y-x

secs=delta_t.seconds+1
#print("First sd guy")
#print(sd.query_devices())
#sd.default.device = 1
#print(sd.query_devices())


def proc1(start_time):
    while time.time() <= start_time:
        pass
    print("Started Proc1 recording")

    #print(sd.query_devices())

    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    #print(myrecording)
    sd.wait()
    print("Done proc1")
    sf.write("test_rec1.wav", data=myrecording, samplerate=fs)



def start_recording(secs):
    start_time = time.time() + secs
    threading.Thread(target=proc1, args=(start_time,)).start()
    #threading.Thread(target=proc2, args=(start_time,)).start()

#t = Timer(secs, start_recording(secs))
#t.start()
start_recording(secs)
for i in range(secs,0,-1):
    print("Seconds left before it begins", i)
    time.sleep(1)

