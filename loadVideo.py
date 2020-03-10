import cv2
import numpy as np
import sys
from random import seed
from random import randint

def check_if_unique(color, current_color_channel):
    if color not in current_color_channel:
        current_color_channel.append(color)

def update_background_model(previous, current, threshold, ksize):
    bg_model = np.zeros_like(current, dtype=np.uint8)
    height, width = current.shape[:2]
    thresh_diff = ksize * ksize * 0.3
    for i in range(0, height, ksize):
        for j in range(0, width, ksize):
            bound_x = i + ksize
            bound_y = j + ksize
            changed = 0
            if i + ksize > height:
                bound_x = height - i
            if j + ksize > width:
                bound_y = width - j
            for x in range(i, bound_x):
                for y in range(j, bound_y):
                    prev = previous[x][y].astype(np.float32)
                    temp = current[x][y].astype(np.float32)
                    difference = abs(prev - temp)
                    if difference > threshold:
                        #bg_model[i][j] = previous[i][j]
                        changed += 1
            if changed > thresh_diff:
                for q in range(i, bound_x, 1):
                    for r in range(j, bound_y, 1):
                        bg_model[q][r] = previous[q][r]
            else:
                for q in range(i, bound_x, 1):
                    for r in range(j, bound_y, 1):
                        bg_model[q][r] = current[q][r]
                    #else:
                        #bg_model[i][j] = current[i][j]
    return bg_model

def median_filter_over_time(first_twenty_frames):
    # instead of checking each pixel, check an area of pixels. Randomly select 3, average in each frame, and find median
    length = len(first_twenty_frames)
    height = first_twenty_frames[0].shape[0]
    width = first_twenty_frames[0].shape[1]
    background_model = np.zeros_like(first_twenty_frames[0], dtype=np.uint8)
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

def med_over_time_kernel(first_twenty_frames, kernel):
    seed(1)
    length = len(first_twenty_frames)
    height, width = first_twenty_frames[0].shape[:2]
    background_model = np.zeros_like(first_twenty_frames[0], dtype=np.float32)
    for i in range(0, height, kernel):
        for j in range(0, width, kernel):
            pixels = []
            medians = []
            for k in range(0, 5):
                pixels.append((randint(i, i + kernel), randint(i, i + kernel)))
            for k in range(0, 5):
                r = g = b = []
                y = pixels[k][0]
                x = pixels[k][1]
                for current_frame in range(0, length):
                    # Append the current pixel
                    check_if_unique(first_twenty_frames[current_frame][y][x][0], r)
                    check_if_unique(first_twenty_frames[current_frame][y][x][1], g)
                    check_if_unique(first_twenty_frames[current_frame][y][x][2], b)
                # Sort each channel array
                r = np.sort(r)
                g = np.sort(g)
                b = np.sort(b)
                # Assign each channel of the background model to the median of the channel
                medians.append((r[int(len(r)/2)],g[int(len(g)/2)],b[int(len(b)/2)]))
            medians.sort()
            selected_median = medians[2]
            for m in range(i, kernel and height):
                for n in range(j, kernel and width):
                    for color in range(0, 3):
                        background_model[m][n][color] = selected_median[color]
    return background_model

def diff(bg_model, gray, thresh, kernel):
    height, width = gray.shape[:2]
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) ^^ line above was frame instead of gray
    #gray = cv2.medianBlur(gray, 5)
    # Increase contrast
    #gray = gray[:,:] * 1.5
    thresh_diff = kernel * kernel * 0.3
    mask = np.full_like(gray, 255, dtype=np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            bg = bg_model[i][j].astype(np.float32)
            g = gray[i][j].astype(np.float32)
            difference = abs(bg - g)
            if difference > thresh:
                mask[i][j] = 0
    ksize = kernel
    for i in range(0, height, ksize):
        for j in range(0, width, ksize):
            #if i - 1 >= 0 and j - 1 >= 0 and i + 1 < height and j + 1 < width: #bounds checking
            adj = 0
            over_thresh = 0
            count = 0
            bound_x = i + ksize
            bound_y = j + ksize
            if bound_x >= height:
                bound_x = height - i
            if bound_y >= width:
                bound_y = width - j
            for x in range(i, bound_x):
                for y in range(j, bound_y):
                    if mask[x][y] == 0:
                        over_thresh += 1
                    elif (y-bound_y >= 0 and x-bound_x >= 0) and (mask[x][y-bound_y] == 0 or mask[x-bound_x][y] == 0):
                        if gray[x][y] > thresh:
                            adj += 1
                    count += 1
            if count != 0: #and total != 0:
                if over_thresh/count > 0.4 or adj/count > 0.2: #and is_most/total > 0.8:
                    continue
                else:
                    val = 255
                    for x in range(i, bound_x):
                        for y in range(j, bound_y):
                            mask[x][y] = val

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
    frame = cv2.medianBlur(frame, 9)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for i in range(0, height, kernel):
        for j in range(0, width, kernel):
            changed = []
            if i + kernel >= width:
                kernel = width-i
            if j + kernel >= height:
                kernel = height-j
            for k in range(i, i+kernel):
                for l in range(j, j+kernel):
                    bg_mod_pixel = bg_model[k][l].astype(np.float32)
                    current_gray_pixel = gray[k][l].astype(np.float32)
                    if abs(bg_mod_pixel - current_gray_pixel) > threshold:
                        changed.append([i+k,j+l])
            length_of_changed_pixels = len(changed)
            kernel_size = kernel*kernel
            if length_of_changed_pixels > 0:
                percent_changed = length_of_changed_pixels/kernel_size
                if percent_changed >= 0.5:
                    for y in range(i, i+kernel):
                        for x in range(j, j+kernel):
                            gray[y][x] = 0
                else:
                    for y in range(i, i+kernel):
                        for x in range(j, j+kernel):
                            gray[y][x] = bg_model[y][x]
                '''elif percent_changed >= 0.5:
                    for m in range(0, length_of_changed_pixels):
                        y = changed[m][0]
                        x = changed[m][1]
                        gray[y][x] = 0'''
    return gray

'''def remove_noise(frame, kernel):
    height, width = frame.shape[:2]
    for i in range (0, height):
        for j in range(0, width):'''

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
    frame_width = int(cap.get(3) * 0.5)
    frame_height = int(cap.get(4) * 0.5)

    # Define the codec and create VideoWriter object.The output is stored in 'output.avi' file.
    out = cv2.VideoWriter('/Users/ktsutter/Downloads/car_edited.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_height, frame_width))

    num_frames_for_model = 20
    threshold = 15
    kernel = 10
    frames_for_bg_model = []
    for i in range(0, num_frames_for_model):
        read_correctly, frame = cap.read()
        if read_correctly:
            frame = cv2.resize(frame, (frame_width, frame_height), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            frames_for_bg_model.append(frame)
        else:
            break
    bg_model = median_filter_over_time(frames_for_bg_model)
    bg_model = cv2.medianBlur(bg_model, 9)
    # Increase contrast
    #bg_model = bg_model[:, :, :] * 1.3
    # Convert to grayscale
    bg_model = cv2.cvtColor(bg_model, cv2.COLOR_BGR2GRAY)
    height, width = bg_model.shape[:2]
    frames_since_movement = np.zeros_like(bg_model, np.uint8)
    #^^check to see if current pixel hasn't moved (in flagged function)\
    #if its change is 0, increment array at index of pixel by one.
    #if value at index of pixel/num frames visited so far > some threshold (0.5?), ignore current index
    #if greater than threshold, analyze. Then, if change is > threshold, set to black.
    # OR an array with two values at each index: one is time since last change (set to black) and the other is
    #total number of times that the pixel has been outside range. When creating new bg_model, ignore if time since last
    #change is high enough
    while frames_to_read:
        #read_correctly, current_frame = cap.read()
        if read_correctly: # or num_frames > 0:
            current_frames = []
            for i in range(0, num_frames_for_model):
                read_correctly, current_frame = cap.read()
                if read_correctly:
                    current_frame = cv2.resize(current_frame, (frame_width, frame_height), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
                    current_frames.append(current_frame)
                else:
                    break
            length = len(current_frames)
            for i in range(0, length):
                gray = flag_as_changed(current_frames[i], bg_model, kernel, threshold)
                bg_mask = diff(bg_model, gray, threshold, kernel)
                # Copy the resulting mask to the current frame
                result = cv2.copyTo(current_frames[i], mask=bg_mask)
                result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
                # Save frame to file
                out.write(result)
            potential_bg_model = median_filter_over_time(current_frames)
            potential_bg_model = cv2.cvtColor(potential_bg_model, cv2.COLOR_BGR2GRAY)
            bg_model = update_background_model(bg_model, potential_bg_model, threshold*2, kernel)

    # Release video capture and write objects and close all frames
    cap.release()
    out.release()

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
    return 0

__main()