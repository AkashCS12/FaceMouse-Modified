# Importing packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from pyautogui import size, scroll
import time
import mediapipe as mp
import cv2
import mouse
import threading
import math
from collections import deque
from tkinter import Tk, Label

# Initializing indexes for the features to track as an Ordered Dictionary
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("right_eye", [33, 160, 158, 133, 153, 144]),
    ("left_eye", [362, 385, 387, 263, 373, 380]),
    ("nose", [1, 2, 98, 327, 168, 122, 6, 197, 195, 5]),  # approximate landmarks for nose tip
    ("mouth", [61, 146, 146, 91, 91, 181, 181, 84, 84, 17, 17, 314, 314, 405, 405, 321,
               321, 375, 375, 291, 61, 185, 185, 40, 40, 39, 39, 37, 37, 0, 0, 267, 267,
               269, 269, 270, 270, 409, 409, 291, 78, 95, 95, 88, 88, 178, 178, 87, 87, 14,
               14, 317, 317, 402, 402, 318, 318, 324, 324, 308, 78, 191, 191, 80, 80, 81,
               81, 82, 82, 13, 13, 312, 312, 311, 311, 310, 310, 415, 415, 308])  # approximate landmarks for mouth
])

# Mediapipe initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

def shape_arr_func(results, dtype="int"):
    face_landmarks = results.multi_face_landmarks[0].landmark
    coords = np.zeros((468, 2), dtype=dtype)
    for i in range(0, 468):
        coords[i] = (int(face_landmarks[i].x * frame_width), int(face_landmarks[i].y * frame_height))
    return coords

def mvmt_func(x):
    if x > 1.:
        return math.pow(x, 3.0 / 2.0)
    elif x < -1.:
        return -math.pow(abs(x), 3.0 / 2.0)
    elif 0. < x < 1.:
        return 1.0
    elif -1. < x < 0.:
        return -1.0
    else:
        return 0.0

def ear_func(eye):
    v1 = dist.euclidean(eye[1], eye[5])
    v2 = dist.euclidean(eye[2], eye[4])
    h = dist.euclidean(eye[0], eye[3])
    ear = (v1 + v2) / (2.0 * h)
    return ear

# Constants for blink detection
EYE_AR_THRESH = 0.15
EYE_AR_CONSEC_FRAMES_MIN = 3
EYE_AR_CONSEC_FRAMES_MAX = 5

COUNTER = 0
TOTAL = 0
BLINK_DETECTED = False

RIGHT_EYE_COUNTER = 0
RIGHT_BLINK_DETECTED = False

# Constants for mouth open detection
MOUTH_OPEN_CONSEC_FRAMES_MIN = 3
MOUTH_OPEN_CONSEC_FRAMES_MAX = 5

MOUTH_COUNTER = 0
MOUTH_OPEN_DETECTED = False

TOGGLE_COOLDOWN = 2  # 2 seconds cooldown for mouth open drag toggle
last_toggle_time = time.time()

isMouseDown = False
click_type = ""
mouse_control_enabled = True

vs = cv2.VideoCapture(0)
time.sleep(1.0)

def left_click_func():
    global isMouseDown
    global click_type
    global TOTAL
    if isMouseDown:
        mouse.press(button='left')
        click_type = "Mouse Down"
    else:
        mouse.release(button='left')
        click_type = "Mouse Up"
    
    if TOTAL == 1:
        mouse.click(button='left')
        click_type = "Single Click"
    elif TOTAL == 2:
        mouse.double_click(button='left')
        click_type = "Double Click"
        
    TOTAL = 0

def right_click_func():
    global click_type
    global TOTAL
    mouse.click(button='right')
    click_type = 'Right Click'
    TOTAL = 0

sclFact = 5
firstRun = True
scrolling_up = False
scrolling_down = False

global xC
global yC

mouse.move(size()[0] // 2, size()[1] // 2)

def track_nose(nose):
    global xC
    global yC
    global firstRun
    cx = nose[3][0]
    cy = nose[3][1]
    if firstRun:
        xC = cx
        yC = cy
        firstRun = False
    else:
        xC = cx - xC
        yC = cy - yC
        mouse.move(mvmt_func(-xC) * sclFact, mvmt_func(yC) * sclFact, absolute=False, duration=0)
        xC = cx
        yC = cy

def toggle_mouse_control():
    global mouse_control_enabled
    global last_toggle_time
    current_time = time.time()
    if current_time - last_toggle_time >= TOGGLE_COOLDOWN:
        mouse_control_enabled = not mouse_control_enabled
        last_toggle_time = current_time
        print(f"Mouse control enabled: {mouse_control_enabled}")

def continuous_scroll():
    while True:
        if scrolling_up:
            scroll(-15)  # Increase the value to scroll more lines at once
            time.sleep(0.05)  # Reduce the sleep duration for faster scrolling
        elif scrolling_down:
            scroll(15)  # Increase the value to scroll more lines at once
            time.sleep(0.05)  # Reduce the sleep duration for faster scrolling
        else:
            time.sleep(0.1)  # This sleep duration remains the same when not scrolling

scroll_thread = threading.Thread(target=continuous_scroll, daemon=True)
scroll_thread.start()

def track_head_movement(nose):
    global yC
    global scrolling_up
    global scrolling_down

    cx = nose[3][0]
    cy = nose[3][1]
    
    if firstRun:
        yC = cy
    else:
        delta_y = cy - yC
        if delta_y < -20:  # Head moved up significantly
            scrolling_up = True
            scrolling_down = False
        elif delta_y > 20:  # Head moved down significantly
            scrolling_up = False
            scrolling_down = True
        else:  # Head is in the middle position
            scrolling_up = False
            scrolling_down = False

# Initialize MAR history buffer for smoothing
mar_history = deque(maxlen=10)

def smooth_mar(mar):
    mar_history.append(mar)
    # Use the median for smoothing to reduce impact of outliers
    return np.median(mar_history)

def check_mouth_open(mouth):
    global MOUTH_COUNTER
    global MOUTH_OPEN_DETECTED

    # Define landmarks for the mouth
    upper_lip = [mouth[i] for i in [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]]
    lower_lip = [mouth[i] for i in [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73]]

    # Calculate the vertical distance between the upper and lower lips
    upper_lip_center = np.mean(upper_lip, axis=0)
    lower_lip_center = np.mean(lower_lip, axis=0)
    vertical_distance = dist.euclidean(upper_lip_center, lower_lip_center)

    # Calculate the horizontal distance between the corners of the mouth
    left_corner = mouth[48]
    right_corner = mouth[54]
    horizontal_distance = dist.euclidean(left_corner, right_corner)

    # Calculate MAR
    mar = vertical_distance / horizontal_distance if horizontal_distance > 0 else 0

    # Smooth MAR value
    mar = smooth_mar(mar)

    # Define the MAR threshold
    MAR_THRESHOLD = 0.9  # Adjust this threshold based on your observations

    # Check if MAR indicates an open mouth
    if mar > MAR_THRESHOLD:
        MOUTH_COUNTER += 1
        if MOUTH_COUNTER >= MOUTH_OPEN_CONSEC_FRAMES_MIN:
            if not MOUTH_OPEN_DETECTED:
                toggle_mouse_down()
                MOUTH_OPEN_DETECTED = True
    else:
        if MOUTH_COUNTER >= MOUTH_OPEN_CONSEC_FRAMES_MAX:
            MOUTH_OPEN_DETECTED = False
        MOUTH_COUNTER = 0
        
def toggle_mouse_down():
    global isMouseDown
    global last_toggle_time
    current_time = time.time()
    if current_time - last_toggle_time >= TOGGLE_COOLDOWN:
        isMouseDown = not isMouseDown
        last_toggle_time = current_time
        status = "on" if isMouseDown else "off"
        print(f"Mouse down (drag) toggle: {status}")
        # Ensure that the mouse state is updated accordingly
        if isMouseDown:
            mouse.press(button='left')
        else:
            mouse.release(button='left')

# Overlay function to display mouse control status
def create_overlay():
    root = Tk()
    root.overrideredirect(True)
    root.geometry(f"300x100+{root.winfo_screenwidth() - 300}+0")  # Position on the top right
    root.wm_attributes("-topmost", 1)
    
    label = Label(root, text="Mouse: ON\nMouse Down: OFF", font=("Helvetica", 16), fg="green", bg="white")
    label.pack()

    def update_overlay():
        if mouse_control_enabled:
            label.config(text="Mouse: ON\nMouse Down: " + ("ON" if isMouseDown else "OFF"), fg="green")
        else:
            label.config(text="Mouse: OFF\nMouse Down: " + ("OFF" if isMouseDown else "OFF"), fg="red")
        root.after(100, update_overlay)

    update_overlay()
    root.mainloop()

# Start overlay in a separate thread
overlay_thread = threading.Thread(target=create_overlay, daemon=True)
overlay_thread.start()

# Main loop
while True:
    ret, frame = vs.read()
    frame_height, frame_width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    timer = cv2.getTickCount()  # Start timer

    if results.multi_face_landmarks:
        shape = shape_arr_func(results)
        leftEye = shape[FACIAL_LANDMARKS_IDXS["left_eye"]]
        rightEye = shape[FACIAL_LANDMARKS_IDXS["right_eye"]]
        nose = shape[FACIAL_LANDMARKS_IDXS["nose"]]
        mouth = shape[FACIAL_LANDMARKS_IDXS["mouth"]]

        leftEAR = ear_func(leftEye)
        rightEAR = ear_func(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if mouse_control_enabled:
            track_nose(nose)
        else:
            track_head_movement(nose)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        noseHull = cv2.convexHull(nose)
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 255), 1)

        # Calculate FPS
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (450, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(frame, "Click Type: {}".format(click_type), (10, 60),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
        cv2.putText(frame, "Mouse Control: {}".format(mouse_control_enabled), (10, 90),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
        cv2.putText(frame, "Mouse Down Toggle: {}".format("ON" if isMouseDown else "OFF"), (10, 120),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(frame, "FPS: {}".format(round(fps, 2)), (10, 150),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        # Display mouth open/close status
        mar = 0
        if results.multi_face_landmarks:
            A = dist.euclidean(mouth[2], mouth[6])
            B = dist.euclidean(mouth[3], mouth[5])
            C = dist.euclidean(mouth[0], mouth[4])
            mar = (A + B) / (2.0 * C)

        cv2.putText(frame, "Mouth MAR: {:.2f}".format(mar), (10, 180),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(frame, "Mouth Open: {}".format("OPEN" if MOUTH_OPEN_DETECTED else "CLOSED"), (10, 210),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        if leftEAR < EYE_AR_THRESH and rightEAR < EYE_AR_THRESH:
            COUNTER += 1
            BLINK_DETECTED = False
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES_MIN:
                if not BLINK_DETECTED:
                    TOTAL += 1
                    BLINK_DETECTED = True
                    if COUNTER <= EYE_AR_CONSEC_FRAMES_MAX:
                        threading.Timer(0.7, left_click_func).start()
                    else:
                        threading.Timer(0.1, right_click_func).start()
            COUNTER = 0

        if rightEAR < EYE_AR_THRESH and leftEAR >= EYE_AR_THRESH:
            RIGHT_EYE_COUNTER += 1
            RIGHT_BLINK_DETECTED = False
        else:
            if RIGHT_EYE_COUNTER >= 1:
                if not RIGHT_BLINK_DETECTED and leftEAR >= EYE_AR_THRESH and RIGHT_EYE_COUNTER >= 30:
                    toggle_mouse_control()
                    RIGHT_BLINK_DETECTED = True
            RIGHT_EYE_COUNTER = 0

        check_mouth_open(mouth)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vs.release()