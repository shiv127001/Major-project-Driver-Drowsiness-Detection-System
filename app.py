import os
import cv2
import streamlit as st
import dlib
import numpy as np
from imutils import face_utils
import sounddevice as sd
import threading
import pandas as pd
from datetime import datetime

# Title of the Streamlit app
st.title('Driver Drowsiness Detection')

# Ask for user's name and vehicle number
name = st.text_input('Enter your name')
vehicle_no = st.text_input('Enter your vehicle number')

# Placeholder to display the current status on top
status_placeholder = st.empty()

# Initialize the video capture object (0 represents the default webcam)
cap = cv2.VideoCapture(0)

# Initialize the face detector and facial landmarks predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define status variables for the blink detection
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
alarm_active = False  # Flag to control the alarm sound

# Initialize or load the Excel file to save drowsiness incidents
excel_file = 'drowsiness_log.xlsx'
if os.path.exists(excel_file):
    df = pd.read_excel(excel_file, index_col=0)
else:
    df = pd.DataFrame(columns=['Name', 'Vehicle Number', 'Time', 'Status'])

# Function to compute Euclidean distance
def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

# Function to calculate the Eye Aspect Ratio (EAR)
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    if ratio > 0.25:
        return 2
    elif 0.21 <= ratio <= 0.25:
        return 1
    else:
        return 0

# Play alarm sound when sleeping state is detected
def play_alarm():
    def alarm_sound():
        duration = 2  # Set the duration of the alarm sound
        fs = 44100  # Set the sampling frequency
        t = np.linspace(0, duration, int(fs * duration), False)  # Generate time vector
        data = np.sin(2 * np.pi * 440 * t)  # Generate sine wave data
        sd.play(data, samplerate=fs, blocking=True)  # Play the sound

    global alarm_active
    if not alarm_active:
        alarm_active = True
        # Start a separate thread to play the alarm sound
        alarm_thread = threading.Thread(target=alarm_sound)
        alarm_thread.start()
        alarm_thread.join()  # Wait for the alarm to finish
        alarm_active = False

# Save drowsiness incident to Excel file
def save_drowsiness_incident(name, vehicle_no, status):
    global df, excel_file
    new_entry = pd.DataFrame([{'Name': name, 'Vehicle Number': vehicle_no, 'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Status': status}])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_excel(excel_file, index=True)

# Streamlit function to display video frames
def display_video():
    global sleep, drowsy, active, status, color

    # Placeholder for video frames
    col1, col2 = st.columns(2)
    video_placeholder = col1.empty()
    face_frame_placeholder = col2.empty()
    
    # Stop button to exit the loop
    stop_button = st.button('Stop', key='stop_button')

    while True:
        ret, frame = cap.read()
        
        if not ret:
            st.error("Error: Failed to read frame.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = detector(gray)
        face_frame = frame.copy()
        
        # Process each detected face
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            # Draw rectangles around detected faces
            cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Get facial landmarks
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            # Process eye blinks
            left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            # Judge the eye blinks
            if left_blink == 0 or right_blink == 0:
                sleep += 1
                drowsy = 0
                active = 0
                if sleep > 6:
                    status = "--SLEEPING--"
                    color = (255, 0, 0)
                    play_alarm()
                    save_drowsiness_incident(name, vehicle_no, status)
            elif left_blink == 1 or right_blink == 1:
                sleep = 0
                active = 0
                drowsy += 1
                if drowsy > 6:
                    status = "-Drowsy-"
                    color = (0, 0, 255)
                    save_drowsiness_incident(name, vehicle_no, status)
            else:
                drowsy = 0
                sleep = 0
                active += 1
                if active > 6:
                    status = "Active"
                    color = (0, 255, 0)
            
            # Display status on the frame
            cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Draw facial landmarks on the face frame
            for n in range(0, 68):
                (x, y) = landmarks[n]
                cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)
        
        # Convert the frames to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)

        # Display the frames in the Streamlit app side by side
        video_placeholder.image(frame, caption='Live Video Feed', use_column_width=True)
        face_frame_placeholder.image(face_frame, caption='Face Detection and Landmarks', use_column_width=True)
        
        # Display the current status in the Streamlit app
        status_placeholder.markdown(f"**Status:** {status}")

        # Check if stop button is pressed
        if stop_button:
            break

# Run the Streamlit app
if __name__ == '__main__':
    if name and vehicle_no:  # Check if the user has entered the required information
        display_video()
    else:
        st.warning('Please enter your name and vehicle number.')

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
