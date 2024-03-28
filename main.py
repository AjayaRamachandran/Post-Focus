###### IMPORT ######
from PIL import Image
import numpy as np
import random as rand
from math import *

# Open the image file
image = Image.open('images/image-small.png')
depth = Image.open('images/depthmap-small.png')

# Convert the image to a NumPy array
imageArray = np.array(image)
depthArray = np.array(depth)

# Close the image
image.close()
depth.close()

print(imageArray.shape[0])
print(depthArray.shape)

###### FUNCTIONS ######
def generateKernel(size, x, y):
    kernel = np.ndarray(shape=(size*2, size*2), dtype=np.float16)
    #print(kernel.shape)
    centerX, centerY = max(min(x, imageArray.shape[0] - 1), 0), max(min(y, imageArray.shape[1] - 1), 0)
    totalSquares = 0
    for rowNum, row in enumerate(kernel):
        for pixelNum, pixel in enumerate(row):
            imageX, imageY = max(min(x + pixelNum - size, imageArray.shape[0] - 1), 0), max(min(y + rowNum - size, imageArray.shape[1] - 1), 0)
            if sqrt((rowNum - size)**2 + (pixelNum - size)**2) < size and depthArray[imageX][imageY][0] >= depthArray[centerX][centerY][0]: # checks to see if the pixel is within the border of the kernel circle AND that it is not in the foreground
                kernel[rowNum][pixelNum] = 1
                totalSquares += 1
            else:
                kernel[rowNum][pixelNum] = 0
    kernel = kernel / totalSquares
    #print(totalSquares)
    return kernel
    #kernelImage = Image.fromarray((kernel))
    #kernelImage.save("kernel.png")


def returnKernelAverage(x, y):
    size = int(depthArray[x][y][0] / 70) + 1
    kernel = generateKernel(size = size, x=x, y=y)
    weightedAverage = [0,0,0,255]
    for rowNum, row in enumerate(kernel):
        for pixelNum, pixel in enumerate(row):
            imageX, imageY = max(min(x + rowNum - size, imageArray.shape[0] - 1), 0), max(min(y + pixelNum - size, imageArray.shape[1] - 1), 0)
            for channel in range(3):
                weightedAverage[channel] = weightedAverage[channel] + kernel[pixelNum][rowNum] * imageArray[imageX][imageY][channel]
    return weightedAverage


###### MAIN ######

#print(list(enumerate(imageArray)))
for rowNum, row in enumerate(imageArray): # marches downward
    for pixelNum, pixel in enumerate(row): # marches across by pixel
        for channel in range(len(imageArray[rowNum][pixelNum])-1):
            #scalar = 1- depthArray[rowNum][pixelNum][channel]/255
            #imageArray[rowNum][pixelNum][channel] = imageArray[rowNum][pixelNum][channel] * scalar
            imageArray[rowNum][pixelNum] = returnKernelAverage(rowNum, pixelNum)
            #print("pixel calculated.")

newImage = Image.fromarray((imageArray))
newImage.save("output.png")

#generateKernel(20)




'''
            max(min(pixel[0] + (rand.randint(-4, 4) * rand.randint(-4, 4)), 255), 0),
            max(min(pixel[1] + (rand.randint(-4, 4) * rand.randint(-4, 4)), 255), 0),
            max(min(pixel[2] + (rand.randint(-4, 4) * rand.randint(-4, 4)), 255), 0),'''