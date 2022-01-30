import cv2
import numpy as np

cap = cv2.VideoCapture(1)
image = cv2.imread("bk.jpg")

while True:
    ret, frames = cap.read()

    frames = cv2.resize(frames, (640, 480))
    image = cv2.resize(image, (640, 480))

    u_black = np.array([104, 153, 70])
    l_black = np.array([30, 30, 0])

    mask = cv2.inRange(frames, l_black, u_black)
    res = cv2.bitwise_and(frames,frames, mask = mask)

    f = frames - res
    f = np.where(f == 0, image, f)

    cv2.imshow("video", frames)
    cv2.imshow("mask", f) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()



