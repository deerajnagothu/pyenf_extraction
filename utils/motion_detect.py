import numpy as np
import cv2

cap = cv2.VideoCapture('../Recordings/Corridor/resized_corridor_surveillance.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=80 ,detectShadows=True)

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

# I can use fgmask to count the occurences of white pixels (255) or shadow pixels == gray pixels (127)
# np.count_nonzero(fgmask == 127) this should give me how many pixels were modified.
# with this I can setup a threshold of when to do superpixel masking
# Instead of evaluating mask for every frame I can choose to do it after every 10 frames or 20 frames
# depending on the computational requirement.
# With this mask, I can replace the 255 and 127 with 0 and rest all to 1. Masking the necessary
# values with superpixel segmensts. Then masking out all the superpixel region.
# then calculate the superpixel values.