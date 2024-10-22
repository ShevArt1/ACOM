import numpy as np
import cv2

def gauss(x, y, a, b, sigma):
    o = 2 * sigma * sigma
    return 1/(o * np.pi) * np.e **(-1 * ((x-a)* (x-a) +(y-b)* (y-b))/o)

def make_kernel(ksize, sigma):
    kernel = np.zeros((ksize, ksize))
    a = b = ksize//2 + 1
    for i in range(ksize):
        for j in range(ksize):
            kernel[i, j] = gauss(i+1, j+1, a, b, sigma)

    return kernel

def test():
    ar1 = make_kernel(3, 5)
    print("kernel size = ", 3, "deviation = ", 5)
    print(ar1)
    ar2 = make_kernel(5, 5)
    print("kernel size = ", 5, "deviation = ", 5)
    print(ar2)
    ar3 = make_kernel(7, 5)
    print("kernel size = ", 7, "deviation = ", 5)
    print(ar3)

    ar1 /= np.sum(ar1)
    print(ar1)
    print("sum of normalised= ", np.sum(ar1))
    ar2 /= np.sum(ar2)
    print(ar2)
    print("sum of normalised= ", np.sum(ar2))
    ar3 /= np.sum(ar3)
    print(ar3)
    print("sum of normalised= ", np.sum(ar3))



def gaussian_blur(img, ksize, sigma):
    kernel = make_kernel(ksize, sigma)
    # print(kernel)
    # print(np.sum(kernel))

    kernel /= np.sum(kernel)
    # print(kernel)
    # print(np.sum(kernel))

    blurred = img.copy()
    h, w = img.shape[:2]
    half_kernel_size = int(ksize // 2)
    for y in range(half_kernel_size, h - half_kernel_size):
        for x in range(half_kernel_size, w - half_kernel_size):
            val = 0
            for k in range(-(ksize // 2), ksize // 2 + 1):
                for l in range(-(ksize // 2), ksize // 2 + 1):
                    val += img[y + k, x + l] * kernel[k + half_kernel_size, l + half_kernel_size]
            blurred[y, x] = val

    return blurred

#test()

img = cv2.imread('./Images/bRelzN7xP7.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original', img)
cv2.imwrite(f'noblur.jpg', img)

# sigmas =[5, 3, 3, 5]
# sizes =[7, 7, 11, 11]
#
# for i in range(4):
#     ksize= sizes[i]
#     sigma = sigmas[i]
#     img_blur_mine = gaussian_blur(img, ksize, sigma)
#
#     cv2.imshow(f'Blurred (kernel_size={ksize}, std_deviation={sigma})', img_blur_mine)
#     cv2.imwrite(f'blur_{ksize}_{sigma}.jpg', img_blur_mine)


ksize= 11
sigma = 5
img_blur_mine = gaussian_blur(img, ksize, sigma)
cv2.imshow(f'Blurred (kernel_size={ksize}, std_deviation={sigma})', img_blur_mine)
cv2.imwrite(f'blur_{ksize}_{sigma}.jpg', img_blur_mine)

img_blur_lib = cv2.GaussianBlur(img, (11,11), 5)
cv2.imshow('Blurred by library', img_blur_lib)

cv2.imwrite(f'blur_lib_11_5.jpg', img_blur_lib)

cv2.waitKey(0)
cv2.destroyAllWindows()
