import cv2
import numpy as np

def _ycc(r, g, b): # in (0,255) range
    y = .257*r + .504*g + .098*b +16.0
    cb = 128 -.148*r -.291*g - .439*b
    cr = 128 +.439*r - .368*g - .071*b
    return y, cb, cr

img = cv2.imread('dgu_night_color.png', cv2.IMREAD_COLOR)  # input image
height, width = img.shape[0], img.shape[1]
size = height * width  # channel size

img_b, img_g, img_r = cv2.split(img)  # split an input image to r,g,b channel

zeros = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
img_yi = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
img_cb = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
img_cr = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
img_yo = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
img_bo = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
img_go = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
img_ro = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")

# RGB -> YCbCr
for i in range(height):
    for j in range(width):
        img_yi[i][j], img_cb[i][j], img_cr[i][j] = _ycc(img_r[i][j], img_g[i][j], img_b[i][j])

# calculate Y's histogram
histogram = np.zeros(256)
for y in range(height):
    for x in range(width):
        value = img_yi[y, x]
        histogram[value] += 1

# calculate Y's cumulative histogram
cumulative_histogram = np.zeros(256)
sum = 0
for i in range(256):
    sum += histogram[i]
    cumulative_histogram[i] = sum

# normalized cumulative histogram
normalized_cumulative_histogram = np.zeros(256)
for i in range(256):
    normalized_cumulative_histogram[i] = cumulative_histogram[i] / size

# make histogram equalized Y image
for y in range(height):
    for x in range(width):
        img_yo[y, x] = normalized_cumulative_histogram[img_yi[y, x]] * 255

for y in range(height):
    for x in range(width):
        B = img_yo[y][x] * ((img_b[y][x] / img_yi[y][x])** 0.9)
        G = img_yo[y][x] * ((img_g[y][x] / img_yi[y][x])** 0.9)
        R = img_yo[y][x] * ((img_r[y][x] / img_yi[y][x])** 0.9)
        if B > 255:
            B = 255
        if G > 255:
            G = 255
        if  R > 255:
            R = 255
        img_bo[y][x] = B
        img_go[y][x] = G
        img_ro[y][x] = R

result=cv2.merge((img_bo,img_go,img_ro))
cv2.imshow("result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
