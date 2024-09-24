import numpy as np
import cv2

cv2.namedWindow('Result')

lastX = -1
lastY = -1

cap = cv2.VideoCapture(0)

ok, frame = cap.read()
if not ok:
    exit(1)
h, w, c = frame.shape

path = np.zeros((h, w, 3), np.uint8)

while True:
    ok, frame = cap.read()
    key = cv2.waitKey(1) & 0xFF
    if not ok or key == 27:
        break

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
    frame_hsv[:, :, 0] = (frame_hsv[:, :, 0] + 128) % 255

    # h_min = 128-9
    # h_max = 128+9
    # s_min = 140
    # s_max = 255
    # v_min = 3
    # v_max = 255

    h_min = 128 - (255/360*10)
    h_max = 128 + (255/360*10)
    s_min = 255//100*65
    s_max = 255
    v_min = 255//100*65
    v_max = 255

    # h_min = 128-9
    # h_max = 128+9
    # s_min = 100
    # s_max = 255
    # v_min = 100
    # v_max = 255
    hsv_min = np.array((h_min, s_min, v_min))
    hsv_max = np.array((h_max, s_max, v_max))

    mask = cv2.inRange(frame_hsv, hsv_min, hsv_max)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))
    frame_red = cv2.bitwise_and(frame, frame, mask=mask)

    moments = cv2.moments(mask, True)
    dM01 = moments['m01']
    dM10 = moments['m10']
    dArea = moments['m00']
    if dArea > 1000:
        posX = int(dM10 / dArea)
        posY = int(dM01 / dArea)
        cv2.circle(frame, (posX, posY), 10, (255, 0, 100), -1)

        x, y, w, h = cv2.boundingRect(mask)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 100), 2)
        if lastX >= 0 and lastY >= 0:
            cv2.line(path, (lastX, lastY), (posX, posY), (255, 0, 100), 2)
        lastX = posX
        lastY = posY
    else:
        lastX = -1
        lastY = -1
    frame = cv2.add(frame, path)

    cv2.imshow('Result', frame)
    cv2.imshow('Threshold', frame_red)

cv2.destroyAllWindows()
