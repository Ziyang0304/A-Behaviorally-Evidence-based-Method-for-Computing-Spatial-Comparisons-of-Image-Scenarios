import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


# Hash comparison
def cmpHash(hash1, hash2, shape=(10, 10)):
    n = 0
    #If the hash length is different, -1 indicates an error in parameter passing
    if len(hash1) != len(hash2):
        return -1
    # Ergodic judgment
    for i in range(len(hash1)):
        # If equal, n counts +1, and n is ultimately the similarity
        if hash1[i] == hash2[i]:
            n = n + 1
    return n / (shape[0] * shape[1])


# Average hash. It's fast, but often inaccurate.
def aHash(img, shape=(10, 10)):
    # Mean hashing algorithm
    # The scale is 10 by 10
    img = cv2.resize(img, shape)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s is the pixel and the initial value is 0. hash_str is the hash value and the initial value is "".
    hash_str = ''
    # Iterate the sum of pixels
    for i in range(shape[0]):
        for j in range(shape[1]):
            s = s + gray[i, j]
    # Averaging gray
    avg = s / 100
    # The hash value of the image is generated if the gray scale is greater than the average value is 1, but is 0
    for i in range(10):
        for j in range(10):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# Differential hash. The accuracy is high and the speed is very fast
def dHash(img, shape=(10, 10)):
    # Differential hashing algorithm
    # 
    img = cv2.resize(img, (shape[0] + 1, shape[1]))
    # Scale by 10*10
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # The pixel before each row is 1 greater than the pixel after, and the opposite is 0, generating a hash
    for i in range(shape[0]):
        for j in range(shape[1]):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def calculate(image1, image2):
    # Gray histogram algorithm
    # Calculate the similarity value of the histogram of a single channel
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # Calculate the coincidence degree of histogram
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                     (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree


def classify_hist_with_split(image1, image2, size=(256, 256)):
   # RGB Histogram similarity of each channel
    # After resize the image, separate it into three RGB channels, and then calculate the similar value of each channel
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data


def cmpHash(hash1, hash2):
    #Hash value comparison
    # The combination of 1 and 0 in the # algorithm is the fingerprint hash of the image. The order is not fixed, but the comparisons must be in the same order.
    #Compare the fingerprints of two images and calculate the Hamming distance, that is, how many hash values of two 64 bits are different. The smaller the different bits, the more similar the images are
    # Hamming distance: the steps required for one set of binary data to become another set of data can measure the difference between two graphs. The smaller the Hamming distance, the higher the similarity. The Hamming distance is zero, which means the two pictures are exactly the same
    n = 0
    # If the hash length is different, -1 indicates an error in parameter passing
    if len(hash1) != len(hash2):
        return -1
    # Ergodic judgment
    for i in range(len(hash1)):
        # If not equal, n counts +1, and n is finally similarity
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


def getImageByUrl(url):
    # Gets the image object according to the image url
    html = requests.get(url, verify=False)
    image = Image.open(BytesIO(html.content))
    return image


def PILImageToCV():
    # OpenCV image convert to PIL image
    path = "/Users/waldenz/Documents/Work/doc/TestImages/t3.png"
    img = Image.open(path)
    plt.subplot(121)
    plt.imshow(img)
    print(isinstance(img, np.ndarray))
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    print(isinstance(img, np.ndarray))
    plt.subplot(122)
    plt.imshow(img)
    plt.show()


def CVImageToPIL():
    # OpenCV image convert to PIL image
    path = "t1.png"
    img = cv2.imread(path)
    # cv2.imshow("OpenCV",img)
    plt.subplot(121)
    plt.imshow(img)

    img2 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()


def bytes_to_cvimage(filebytes):
    # image byte stream convert to cv image
    image = Image.open(filebytes)
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return img


def pHash(imgfile):
    img_list = []
    # Load and adjust image to 32x32 grayscale image
    img = cv2.imread(imgfile, 0)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)

    # Create a two-dimensional list
    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img  # 填充数据

    # Two-dimensional Dct transform
    vis1 = cv2.dct(cv2.dct(vis0))
    # cv.SaveImage('a.jpg',cv.fromarray(vis0)) 
    vis1.resize(32, 32)

    # Change the two-dimensional list into a one-dimensional list
    img_list = vis1.flatten()

    # Calculate the mean
    avg = sum(img_list) * 1. / len(img_list)
    avg_list = ['0' if i > avg else '1' for i in img_list]

    # gets the hash value
    return ''.join([
        '%x' % int(''.join(avg_list[x:x + 4]), 2)
        for x in range(0, 32 * 32, 4)
    ])


def hammingDist(s1, s2):
    # assert len(s1) == len(s2)
    return 1 - sum([ch1 != ch2
                    for ch1, ch2 in zip(s1, s2)]) * 1. / (32 * 32 / 4)


def runAllImageSimilaryFun(para1, para2):
    # The smaller the value of the three algorithms, mean value, difference value and perception hash algorithm, the more similar they are. The same image value is 0
    # Between the three-histogram algorithm and the single-channel histogram 0-1, the larger the value, the more similar. The same picture is 1
    if para1.startswith("http"):
        img1 = getImageByUrl(para1)
        img1 = cv2.cvtColor(np.asarray(img1), cv2.COLOR_RGB2BGR)

        img2 = getImageByUrl(para2)
        img2 = cv2.cvtColor(np.asarray(img2), cv2.COLOR_RGB2BGR)
    else:
        img1 = cv2.imread(para1)
        img2 = cv2.imread(para2)

    img1 = cv2.imread(
        ''
    )
    img2 = cv2.imread(
        ''
    )

    hash1 = aHash(img1)
    hash2 = aHash(img2)
    n1 = cmpHash(hash1, hash2)
    print('aHash：', n1)

    hash1 = dHash(img1)
    hash2 = dHash(img2)
    n2 = cmpHash(hash1, hash2)
    print('dHash：', n2)

    hash1 = dHash(img1)
    hash2 = dHash(img2)
    n6 = hammingDist(hash1, hash2)
    print('pHash:', n6)

    n4 = classify_hist_with_split(img1, img2)
    print('RGB Hist：', n4)


    plt.subplot(121)
    plt.imshow(Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)))
    plt.subplot(122)
    plt.imshow(Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)))
    plt.show()


if __name__ == "__main__":
    p1 = ""
    p2 = ""
    runAllImageSimilaryFun(p1, p2)
