# importing the required libraries
import serial
import numpy as np
import cv2
from CentroidTracker import CentroidTracker
from trackableobject import TrackableObject


# some utility functions
def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))


def count_within_range(list1, l, r):
    """
    Helper function to count how many numbers in list1 falls into range [l,r]
    """
    c = 0
    # traverse in the list1
    for x in list1:
        # condition check
        if (x >= l) and (x <= r):
            c += 1
    return c


total_up = 0
total_down = 0

# initialize centroid tracker
ct = CentroidTracker()

# a dictionary to map each unique object ID to a TrackableObject
trackableObjects = {}

# height and width of the display screen
height = 540
width = 540

# Initialise the params
params = cv2.SimpleBlobDetector_Params()

# Initialising the parameter values
params.minThreshold = 0
params.maxThreshold = 255

params.filterByArea = True
params.minArea = 3500
params.maxArea = 10000

params.filterByColor = False
params.blobColor = 0

params.filterByCircularity = True
params.minCircularity = 0.5
params.maxCircularity = 1

params.filterByConvexity = True
params.minConvexity = 0.5
params.maxConvexity = 1

params.filterByInertia = True
params.minInertiaRatio = 0.01

window_name = 'Image'
start_point = (0, 270)
end_point = (540, 270)
color = (0, 0, 255)
thickness = 2

ser = serial.Serial(port="COM3", baudrate=115200, stopbits=1)
l1 = []
dk = False
q = 0

while True:

    receive = ser.readline().decode('ascii')

    if receive == "---PRINT PIXEL TEMPERATURE---\n":
        continue

    if receive == "pixel temperature (dK)\n":
        dk = True

        if len(l1) == 32:
            q += 1
            s = np.zeros((32, 32), dtype=np.uint8)
            for i in range(len(l1)):
                for j in range(len(l1[i])):
                    if (l1[i][j] >= 3000) and (l1[i][j] < 3300):
                        s[i][j] = 255
                    else:
                        s[i][j] = 0

            # Removing the noises

            # find all of the connected components (white blobs in your image).
            # im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
            nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(s)

            # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information.
            # here, we're interested only in the size of the blobs, contained in the last column of stats.
            sizes = stats[:, -1]

            # the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
            # you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below.
            sizes = sizes[1:]
            nb_blobs -= 1

            # minimum size of particles we want to keep (number of pixels).
            # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever.
            min_size = 50

            # output image with only the kept components
            im_result = np.zeros((32, 32), dtype=np.uint8)

            # for every component in the image, keep it only if it's above min_size
            for blob in range(nb_blobs):
                if sizes[blob] >= min_size:
                    # see description of im_with_separated_blobs above
                    im_result[im_with_separated_blobs == blob + 1] = 255

            # Image Smoothing Operation
            blur = ((3, 3), 1)
            erode_ = (3, 3)
            dilate_ = (2, 2)
            u = cv2.dilate(cv2.erode(cv2.GaussianBlur(im_result / 255, blur[0], blur[1]), np.ones(erode_)),
                           np.ones(dilate_)) * 255

            ima = cv2.resize(u, dsize=(540, 540))
            v = np.array(ima, dtype=np.uint8)

            l1.clear()
            dk = False

            # Detect blobs
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(v)
            img_with_keypoints = cv2.drawKeypoints(v, keypoints, np.array([]), (0, 0, 255),
                                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # update our centroid tracker using the detected centroids
            objects = ct.update(keypoints)

            # loop over the tracked objects
            for (objectID, centroid) in objects.items():

                # check to see if a trackable object exists for the current object ID
                to = trackableObjects.get(objectID, None)

                # if there is no existing trackable object, create one
                if to is None:
                    to = TrackableObject(objectID, centroid)

                # otherwise, there is a trackable object so we can utilize it to determine direction
                else:
                    # the difference between the y-coordinate of the current
                    # centroid and the mean of previous centroids will tell
                    # us in which direction the object is moving (negative for
                    # 'up' and positive for 'down')
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean
                    to.centroids.append(centroid)

                    # check to see if the object has been counted or not
                    if not to.counted:
                        # if the direction is negative (indicating the object
                        # is moving up) AND the centroid is above the center
                        # line, count the object
                        # the historical centroid must present in the lower half of the screen

                        if direction < 0 and centroid[1] < height // 2 and count_within_range(y, height // 2,
                                                                                              height) > 0:
                            total_up += 1
                            to.counted = True

                        # if the direction is positive (indicating the object is moving down)
                        # AND the centroid is below the center line, count the object
                        # the historical centroid must present in the upper half of the screen
                        elif direction > 0 and centroid[1] > height // 2 and count_within_range(y, 0, height // 2) > 0:

                            total_down += 1
                            to.counted = True

                # store the trackable object in our dictionary
                trackableObjects[objectID] = to

            # Create the central line
            image = cv2.line(img_with_keypoints, start_point, end_point, color, thickness)

            # Create background rectangle with color
            x, y, w, h = 0, 0, 90, 60
            cv2.rectangle(image, (x, 2), (x + w, y + h), (100, 100, 100), -1)

            # Printing the text on the screen { people count of coming IN and going OUT }
            cv2.putText(img=image, text='IN : ' + str(total_up), org=(5, 20), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=.6, color=(0, 255, 0), thickness=1)
            cv2.putText(img=image, text='OUT : ' + str(total_down), org=(5, 45), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=.6, color=(0, 0, 255), thickness=1)

            # Display the screen
            cv2.imshow("Pedestrian Tracker", image)
            cv2.waitKey(1)

        continue

    if dk == True:
        if receive == "\n":
            continue

        res = receive.rstrip()
        res = res.split("\t")
        a = np.array(res, dtype=np.uint16)
        l1.append(a)

ser.close()