import numpy as np
import cv2

def angle(x, y):
    tg = y/x
    v1 = 0.414
    v2 = 2.414
    if((x>0 and y <0 and tg <-v2)or(x<0 and y<0 and tg>v2)): return 0
    elif(x>0 and y <0 and tg < -v1): return 1
    elif((x>0 and y <0 and tg >-v1)or(x>0 and y>0 and tg<v1)): return 2
    elif(x>0 and y>0 and tg<v2): return 3
    elif((x>0 and y>0 and tg>v2)or(x<0 and y>0 and tg<-v2)): return 4
    elif(x<0 and y>0 and tg<-v1): return 5
    elif(x<0 and y<0 and tg<v2): return 7
    else: return 6

def convolution(img, ker):
    grad = np.zeros_like(img, np.int32)
    h, w = img.shape[:2]
    for x in range(1, w - 1):
        for y in range(1, h - 1):
            val = 0
            for k in range(3):
                for l in range(3):
                    val += img[x + k-1, y + l-1] * ker[k, l]
            grad[x, y] = val

    return grad

def ok(neighbour, a,gr):
    if neighbour>0 and a>=gr:
        return True
    else: return False
def task4(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Original', img)
    img_blur = cv2.GaussianBlur(img, (5,5), 0)
    cv2.imshow('Blur', img_blur)

    xkernel = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    ykernel = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])

    gradX = convolution(img, xkernel)
    gradY = convolution(img, ykernel)

    grad_len = np.sqrt(np.add(np.square(gradX), np.square(gradY)))
    max_grad_len = grad_len.max()

    cv2.imshow('gradients', (grad_len / max_grad_len * 255).astype(np.uint8))

    edges = np.zeros_like(img)
    for x in range(1, edges.shape[0] - 1):
        for y in range(1, edges.shape[1] - 1):
            ang = angle(gradX[x, y], gradY[x, y])
            if ang == 0 or ang == 4:
                neighbor1 = [x - 1, y]
                neighbor2 = [x + 1, y]
            elif ang == 1 or ang == 5:
                neighbor1 = [x - 1, y + 1]
                neighbor2 = [x + 1, y - 1]
            elif ang == 2 or ang == 6:
                neighbor1 = [x, y + 1]
                neighbor2 = [x, y - 1]
            elif ang == 3 or ang == 7:
                neighbor1 = [x + 1, y + 1]
                neighbor2 = [x - 1, y - 1]
            if grad_len[x, y] >= grad_len[neighbor1[0], neighbor1[1]] and grad_len[x, y] > grad_len[neighbor2[0], neighbor2[1]]:
                edges[x, y] = 255

    cv2.imshow('edges_before_double_filtering', edges)

    low_level = max_grad_len // 25
    high_level = max_grad_len // 5
    edges2 = edges
    for x in range(edges2.shape[1]):
        for y in range(edges2.shape[0]):
            if(edges2[x, y]>0):
                if grad_len[x, y] < low_level:
                    edges2[x, y] = 0
                elif grad_len[x, y] < high_level:

                    if not(
                            ok(edges2[x-1][y-1],grad_len[x-1][y-1], high_level)or
                            ok(edges2[x-1][y],grad_len[x-1][y], high_level)or
                            ok(edges2[x-1][y+1],grad_len[x-1][y+1], high_level)or
                            ok(edges2[x][y-1],grad_len[x][y-1], high_level)or
                            ok(edges2[x][y+1],grad_len[x][y+1], high_level)or
                            ok(edges2[x+1][y-1],grad_len[x+1][y-1], high_level)or
                            ok(edges2[x+1][y],grad_len[x+1][y], high_level)or
                            ok(edges2[x+1][y+1],grad_len[x+1][y+1], high_level)):
                        edges2[x, y] = 0


    cv2.imshow('edges_filter', edges2)
    edges_library = cv2.Canny(img, 200, 300)
    cv2.imshow('edges_library', edges_library)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # i = 0
    # j =0
    # while True:
    #     edges_library = cv2.Canny(img, i, j)
    #     cv2.imshow('edges_libra', edges_library)
    #     cv2.waitKey(5)
    #     j+=5
    #     if j==300 and i<300:
    #         j=0
    #         i+=5
    #     elif i==300 and j ==300:
    #         break
    #     if cv2.waitKey(10) & 0xFF == 27:
    #         break
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


task4('./Images/bRelzN7xP7.jpg')
