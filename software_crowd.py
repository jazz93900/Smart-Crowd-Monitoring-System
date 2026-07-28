import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile

# -----------------------------
# CONFIG
# -----------------------------
AREA_SIZE = st.number_input("Enter Area Size (m²)", value=50)
DENSITY_THRESHOLD = st.slider("Density Threshold", 0.1, 2.0, 0.5)

st.title("Smart Crowd Monitoring Dashboard")

# -----------------------------
# UPLOAD VIDEOS
# -----------------------------
entry_file = st.file_uploader("Upload Entry Video", type=["mp4", "avi"])
exit_file = st.file_uploader("Upload Exit Video", type=["mp4", "avi"])

# -----------------------------
# PERSON DETECTOR
# -----------------------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def count_people(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes, _ = hog.detectMultiScale(gray, winStride=(8, 8))
    return len(boxes)

# -----------------------------
# PROCESS VIDEO
# -----------------------------
def process_video(file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())

    cap = cv2.VideoCapture(tfile.name)
    counts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        people = count_people(frame)
        counts.append(people)

    cap.release()
    return counts

# -----------------------------
# MAIN LOGIC
# -----------------------------
if entry_file and exit_file:
    st.subheader("Uploaded Videos")

    st.video(entry_file)
    st.video(exit_file)

    if st.button("Run Analysis"):
        st.write("Processing videos...")

        entry_counts = process_video(entry_file)
        exit_counts = process_video(exit_file)

        current_people = 0
        density_data = []

        for i in range(min(len(entry_counts), len(exit_counts))):
            current_people += entry_counts[i]
            current_people -= exit_counts[i]

            density = current_people / AREA_SIZE
            density_data.append(density)

        # ALERT
        alert = any(d > DENSITY_THRESHOLD for d in density_data)

        # -----------------------------
        # DISPLAY RESULTS
        # -----------------------------
        st.subheader("Results")

        st.write(f"Final People Count: {current_people}")
        st.write(f"Max Density: {max(density_data):.2f}")

        if alert:
            st.error("⚠ Overcrowding Detected!")
        else:
            st.success("✅ Crowd is Safe")

        # -----------------------------
        # GRAPHS
        # -----------------------------
        st.subheader("Graphs")

        fig, ax = plt.subplots(2, 1)

        ax[0].plot(density_data)
        ax[0].set_title("Density Over Time")

        ax[1].plot(entry_counts, label="Entry")
        ax[1].plot(exit_counts, label="Exit")
        ax[1].legend()
        ax[1].set_title("Entry vs Exit")

        st.pyplot(fig)