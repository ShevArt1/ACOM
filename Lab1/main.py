import cv2
import numpy as np

def task2():
    img = cv2.imread('./Images/p9eZGMl7JV.jpg', cv2.IMREAD_REDUCED_COLOR_8)
    img2 = cv2.imread('./Images/638401031714736933.png', cv2.IMREAD_ANYDEPTH)
    img3 = cv2.imread('./Images/WyPkDUqlT7unVXmYmualBQ.webp', cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow('Display window', cv2.WINDOW_NORMAL)
    cv2.imshow('Display window', img)
    cv2.waitKey(0)

    cv2.namedWindow('Display window2', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Display window2', img2)
    cv2.waitKey(0)

    cv2.namedWindow('Display window3', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Display window3', img3)
    cv2.waitKey(0)


def task3():
    cap = cv2.VideoCapture('./Images/VID-20200416-WA0003.mp4', cv2.CAP_ANY)

    while(True):
        ret, frame = cap.read()
        if not(ret):
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cv2.waitKey(0)

    cap = cv2.VideoCapture('./Images/VID-20200416-WA0003.mp4', cv2.CAP_ANY)
    def change(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        return hsv
    def change2(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        return hsv
    def change3(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        return hsv

    array = []
    i = 0
    while(True):
        ret, frame = cap.read()
        if not(ret):
            break
        frame = cv2.resize(frame, (640, 480))
        array.append(frame)

    i = len(array)
    while(i >0):
        i-=1
        if i<45:
            cv2.imshow('frame', change(array[i]))
        elif i<89:
            cv2.imshow('frame', change2(array[i]))
        elif i<133:
            cv2.imshow('frame', change3(array[i]))
        else:
            cv2.imshow('frame', array[i])

        if cv2.waitKey(20) & 0xFF == 27:
            break

def task4():
    video = cv2.VideoCapture('./Images/VID-20200416-WA0003.mp4')
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("Task_4_output.mp4", fourcc, 25, (w, h))
    while True:
        ok, img = video.read()
        if not ok or cv2.waitKey(1) & 0xFF == 27:
            break
        cv2.imshow('video', img)
        video_writer.write(img)
    video.release()
    cv2.destroyAllWindows()

def task5():
    img = cv2.imread('./Images/bRelzN7xP7.jpg', cv2.IMREAD_UNCHANGED)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.namedWindow('BGR', cv2.WINDOW_NORMAL)
    cv2.namedWindow('HSV', cv2.WINDOW_NORMAL)
    cv2.imshow('BGR', img)
    cv2.imshow('HSV', img_hsv)
    cv2.waitKey(0)


def task6():
    cap= cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    offw = (640-260)//2
    offh = (480-280)//2

    while True:
        ret,frame= cap.read()

        mask = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = cv2.rectangle(mask, (0 + offw, 120 + offh), (260 + offw, 160 + offh), (255, 255, 255), -1)
        blur = cv2.stackBlur(frame, (63, 63))
        frame[mask == 255] = blur[mask == 255]

        cv2.rectangle(frame, (0+offw, 120+offh), (260+offw, 160+offh), (255, 0, 155), 3)
        cv2.rectangle(frame, (110+offw, 0+offh), (150+offw, 120+offh), (255, 0, 155), 3)
        cv2.rectangle(frame, (110+offw, 160+offh), (150+offw, 280+offh), (255, 0, 155), 3)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()



def task7():
    cap = cv2.VideoCapture(0)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("Task_7_output.mp4", fourcc, fps, (w, h))

    while True:
        ok, vid = cap.read()
        if not ok or cv2.waitKey(1) & 0xFF == 27:
            break


        cv2.imshow('Recording...', vid)
        video_writer.write(vid)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

def task8():
    cap= cv2.VideoCapture(0)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_x = frame_w // 2
    center_y = frame_h // 2
    offw = (frame_w-260)//2
    offh = (frame_h-280)//2

    while True:
        ret,frame= cap.read()

        center = frame[center_y][center_x]

        if (center[0]>center[1]) and (center[0]>center[2]):
            color = [255, 0, 0]
        elif (center[1]>center[2]):
            color = [0, 255, 0]
        else:
            color = [0, 0, 255]

        cv2.rectangle(frame, (0+offw, 120+offh), (260+offw, 160+offh), color, -1)
        cv2.rectangle(frame, (110+offw, 0+offh), (150+offw, 120+offh), color, -1)
        cv2.rectangle(frame, (110+offw, 160+offh), (150+offw, 280+offh), color, -1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def task9():
    ip_address = '192.168.137.56'
    port = '8080'

    video = cv2.VideoCapture(f"http://{ip_address}:{port}/video")
    while True:
        ok, img = video.read()
        if not ok or cv2.waitKey(1) & 0xFF == 27:
            break
        cv2.imshow(f'Video stream from {ip_address}:{port}', img)
    video.release()
    cv2.destroyAllWindows()

task2()
