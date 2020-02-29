import cv2
import numpy as np
import sys
import argparse
import imutils
import dlib
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject

def check_if_unique(color, current_color_channel):
    if color not in current_color_channel:
        current_color_channel.append(color)

def median_filter_over_time(first_twenty_frames):
    # instead of checking each pixel, check an area of pixels. Randomly select 3, average in each frame, and find median
    length = len(first_twenty_frames)
    height = first_twenty_frames[0].shape[0]
    width = first_twenty_frames[0].shape[1]
    background_model = np.zeros_like(first_twenty_frames[0], dtype=np.float32)
    for i in range(0, height):
        for j in range(0, width):
            background_model[i][j] = [0,0,0]
    for i in range(0, height):
        for j in range(0, width):
            r = []
            g = []
            b = []
            for current_frame in range(0, length):
                # Append the current pixel at i,j
                check_if_unique(first_twenty_frames[current_frame][i][j][0], r)
                check_if_unique(first_twenty_frames[current_frame][i][j][1], g)
                check_if_unique(first_twenty_frames[current_frame][i][j][2], b)
            # Sort each channel array
            r = np.sort(r)
            g = np.sort(g)
            b = np.sort(b)
            # Assign each channel of the background model to the median of the channel
            background_model[i][j][0] = r[int(len(r)/2)]
            background_model[i][j][1] = g[int(len(g)/2)]
            background_model[i][j][2] = b[int(len(b)/2)]
    return background_model

def diff(bg_model, gray, thresh):
    height, width = gray.shape[:2]
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) ^^ line above was frame instead of gray
    gray = cv2.medianBlur(gray, 5)
    # Increase contrast
    #gray = gray[:,:] * 1.5
    mask = np.zeros_like(gray, dtype=np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            difference = abs(bg_model[i,j] - gray[i,j])
            if difference > thresh:
                mask[i][j] = 0
            else:
                mask[i][j] = 255
    return mask

def set_to_black(rects, frame):
    frame_height, frame_width = frame.shape[:2]
    # loop over the tracked objects
    bounding_boxes = len(rects)
    for j in range(0, bounding_boxes):
        start_x = rects[j][0]
        start_y = rects[j][1]
        end_x = rects[j][2]
        end_y = rects[j][3]
        if start_x < 0:
            start_x = 0
        if start_y < 0:
            start_y = 0
        if end_x > frame_width:
            end_x = frame_width
        if end_y > frame_height:
            end_y = frame_height
        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                for color in range(0, 3):
                    frame[x][y][color] = 0
    return frame

def flag_as_changed(frame, bg_model, kernel, threshold):
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for i in range(0, width, kernel):
        for j in range(0, height, kernel):
            changed = []
            if i + kernel >= width:
                kernel = width-i
            if j + kernel >= height:
                kernel = height-j
            for k in range(i, kernel):
                for l in range(j, kernel):
                    if abs(gray[i+k][j+l] - bg_model[i+k][j+l]) > threshold:
                        changed.append([i+k,j+l])
            length_of_changed_pixels = len(changed)
            if length_of_changed_pixels > 0 and kernel/length_of_changed_pixels >= 0.5:
                for m in range(0, length_of_changed_pixels):
                    x = changed[m][0]
                    y = changed[m][1]
                    gray[x][y] = 0
    return gray

# Writes edited footage out to a file
def edit_footage(path_to_vid):

    # Read footage into a video capture object
    cap = cv2.VideoCapture(path_to_vid)

    # Check if video loaded successfully
    frames_to_read = cap.isOpened()

    #If something went wrong, exit function
    if not frames_to_read:
        print("Unable to read camera feed")
        return 0

    # Default resolutions of the frame obtained. Convert from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'output.avi' file.
    out = cv2.VideoWriter('/Users/ktsutter/Downloads/car_edited.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    num_frames_for_model = 20
    threshold = 10
    frames_for_bg_model = []
    for i in range(0, num_frames_for_model):
        read_correctly, frame = cap.read()
        if read_correctly:
            frames_for_bg_model.append(frame)

    bg_model = median_filter_over_time(frames_for_bg_model)
    bg_model = cv2.medianBlur(bg_model, 5)
    # Increase contrast
    bg_model = bg_model[:, :, :] * 1.5
    # Convert to grayscale
    bg_model = cv2.cvtColor(bg_model, cv2.COLOR_BGR2GRAY)
    while frames_to_read:
        read_correctly, current_frame = cap.read()
        if read_correctly: # or num_frames > 0:
            gray = flag_as_changed(current_frame, bg_model, 16, 10)
            bg_mask = diff(bg_model, gray, threshold)
            # Copy the resulting mask to the current frame
            result = cv2.copyTo(current_frame, mask=bg_mask)
            # Save frame to file
            out.write(result)

    # Release video capture and write objects and close all frames
    cap.release()
    out.release()
    cv2.destroyAllWindows()

#MEDIAN FILTER NOT UPDATED:
    '''frames_for_bg_model = []
    for i in range(0,num_frames_for_model):
        read_correctly, frame = cap.read()
        if read_correctly:
            frames_for_bg_model.append(frame)

    bg_model = median_filter_over_time(frames_for_bg_model)
    bg_model = cv2.medianBlur(bg_model, 5)
    # Increase contrast
    bg_model = bg_model[:,:,:] * 1.5
    # Convert to grayscale
    bg_model = cv2.cvtColor(bg_model, cv2.COLOR_BGR2GRAY)'''

#HOG WITH MEDIAN UPDATED BG (BEST SO FAR):
    '''while frames_to_read:
        frames_for_bg_model = []
        frames_to_compare = []
        #smooth = num_frames_for_model - int(num_frames_for_model/5)
        for i in range(0, num_frames_for_model):
            read_correctly, frame = cap.read()
            if read_correctly:
                frame = cv2.resize(frame, dsize=(frame_width,frame_height))
                frames_for_bg_model.append(frame)
                #if i >= smooth:
                #    frames_to_compare.append(frame)
            else:
                break
        num_frames = len(frames_for_bg_model)
        if read_correctly: # or num_frames > 0:
            bg_model = median_filter_over_time(frames_for_bg_model)
            bg_model = cv2.medianBlur(bg_model, 5)
            # Increase contrast
            bg_model = bg_model[:, :, :] * 1.5
            # Convert to grayscale
            bg_model = cv2.cvtColor(bg_model, cv2.COLOR_BGR2GRAY)

            for i in range(0, num_frames):
                current_frame = frames_for_bg_model[i]
                # Get background model by examining first 20 frames and getting the median over time
                bg_mask = diff(bg_model, current_frame, threshold)
            # Rotate so that frame is upright and people can be detected
             current_frame = cv2.rotate(current_frame, cv2.ROTATE_90_CLOCKWISE)
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(16, 16),
                                                      scale=1.09)  # scale of 1.1 best so far

                boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

                for (xA, yA, xB, yB) in boxes:
                    # Display the detected boxes in the frame (in color)
                    cv2.rectangle(current_frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
                # Rotate back to write out to file
                current_frame = cv2.rotate(current_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # Copy the resulting mask to the current frame
            result = cv2.copyTo(current_frame, mask=bg_mask)
            # Save frame to file
            out.write(result)'''

#HISTOGRAM OF GRADIENTS
    '''hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    while frames_to_read:
        read_correctly, frame = cap.read()
        if read_correctly:
            height,width = frame.shape[:2]
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            #frame = cv2.resize(frame, (int(height/2), int(width/2)))
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            boxes, weights = hog.detectMultiScale(gray, winStride=(8,8), padding=(16,16), scale=1.09) #scale of 1.1 best so far

            boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

            for (xA, yA, xB, yB) in boxes:
                # display the detected boxes in the color picture
                cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            out.write(frame)
        else:
            break'''

#VERSION 1:
# increases contrast
# temp = frame[:,:,:] * 2
# med_frame = cv2.medianBlur(frame, 5)
# Apply background subtractor to current frame
# fg_mask = bg_sub.apply(med_frame)
# Perform bitwise not on background mask to create foreground mask
# bg_mask = cv2.bitwise_not(fg_mask)
# frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#EDITS:
'''score, diff = compare_ssim(bg_model, frame_gray, full=True)
diff = (diff * 255).astype("uint8")
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]'''
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)

#MORE EDITS:
# result = cv2.bitwise_and(frame, frame, mask=bg_mask)
# Set up the detector with default parameters.
'''detector = cv2.SimpleBlobDetector()
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
# Detect blobs.
keypoints = detector.detect(gray)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
frame_with_keypoints = cv2.drawKeypoints(gray, keypoints, result)'''

# Display the edited frame, then write out to file
'''cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 900, 900)
cv2.imshow('image', result)
key = cv2.waitKey(30)
if key == 27:
    break'''
#FIRST ATTEMPT:
''''#Takes two arguments (TODO, for now just takes 1) and compares their yuv values, mark each pixel that has changed
#enough to warrant being blacked out, return edited frame
def check_threshold(frame, prev_frame, bg_sub):
    # Convert from BGR to grayscale (opencv reads in images with BGR instead of RGB ordering)
    height = frame.shape[0]
    width = frame.shape[1]
    threshold = 1.5
    # If frames are different dimensions, something has gone very wrong
    assert(prev_frame.shape[0] == height and prev_frame.shape[1] == width)

    # Convert to grayscale
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Apply 5x5 median blur to each of the frames to remove 'salt + pepper' noise
    median = cv2.medianBlur(frame, 5)
    prev_median = cv2.medianBlur(prev_frame, 5)

    # Check to see if the pictures are different by more than threshold: if so, compare and edit the frames to
    # remove any moving elements. Else, continue on to next frame
    #diff = cv2.absdiff(median, prev_median)
    #if diff.any() > threshold:
        #frame = compare_frames(frame, bg_sub)
    print("\nDifference > threshold\n")
    for i in range(0, height):
        for j in range(0, width):
            diff = abs(median[i,j] - prev_median[i,j])
            # If the difference between the previous and current frame is > threshold, black it out
            if diff.any() > threshold:
                frame[i,j][0] = frame[i,j][1] = frame[i,j][2] = 0
    #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame'''


def __main():
    video = sys.argv[1]
    edit_footage(video)

__main()