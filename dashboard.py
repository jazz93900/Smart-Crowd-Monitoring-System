import streamlit as st
import cv2
import pandas as pd
import math
import time
import serial
from ultralytics import YOLO

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Crowd Monitoring",
    layout="wide"
)

# =====================================================
# LOAD YOLO
# =====================================================
model = YOLO("yolov8n.pt")

# =====================================================
# SERIAL (CHANGE COM PORT)
# =====================================================
ser = serial.Serial('COM10', 115200)

# =====================================================
# VIDEO SOURCES (UNCHANGED)
# =====================================================
cap1 = cv2.VideoCapture("zone1.mp4")
cap2 = cv2.VideoCapture("zone2.mp4")

zone1_graph = []
zone2_graph = []

frame_count = 0
ir_count = 0

# =====================================================
# PROCESS FUNCTION (UNCHANGED)
# =====================================================
def process_zone(frame):

    frame = cv2.resize(frame, (700, 400))

    height, width = frame.shape[:2]
    frame_area = width * height

    results = model(frame)[0]
    boxes = results.boxes

    people_count = 0
    occupied_area = 0

    for box in boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == 0 and conf > 0.3:

            people_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            w = x2 - x1
            h = y2 - y1

            occupied_area += w * h

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    density_ratio = occupied_area / frame_area

    if density_ratio > 0.25:
        density = "HIGH"
    elif density_ratio > 0.12:
        density = "MEDIUM"
    else:
        density = "LOW"

    return frame, people_count, density

# =====================================================
# UI LAYOUT (UNCHANGED)
# =====================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Zone 1")
    zone1_video = st.empty()
    zone1_count = st.empty()
    zone1_density = st.empty()
    zone1_chart = st.empty()

with col2:
    st.subheader("📍 Zone 2")
    zone2_video = st.empty()
    zone2_count = st.empty()
    zone2_density = st.empty()
    zone2_chart = st.empty()

st.markdown("---")
total_placeholder = st.empty()
alert_placeholder = st.empty()

# =====================================================
# MAIN LOOP
# =====================================================
while True:

    frame_count += 1

    # ===== READ IR COUNT =====
    if ser.in_waiting:
        try:
            ir_count = int(ser.readline().decode().strip())
        except:
            pass

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1:
        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    if not ret2:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    if frame_count % 2 != 0:
        continue

    frame1, count1, density1 = process_zone(frame1)
    frame2, count2, density2 = process_zone(frame2)

    # ===== USE IR COUNT =====
    total_people = ir_count

    # ===== OVERCROWD LOGIC =====
    overcrowded = False

    if density1 == "HIGH" or density2 == "HIGH":
        overcrowded = True

    if total_people > 20:
        overcrowded = True

    # ===== SEND TO ESP32 =====
    if overcrowded:
        ser.write(b'1')   # buzzer ON (handled in ESP32)
    else:
        ser.write(b'0')   # buzzer OFF

    # ===== GRAPH UPDATE =====
    if frame_count % 20 == 0:

        zone1_graph.append(count1)
        zone2_graph.append(count2)

        if len(zone1_graph) > 20:
            zone1_graph.pop(0)

        if len(zone2_graph) > 20:
            zone2_graph.pop(0)

    # ===== DISPLAY =====
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    zone1_video.image(frame1_rgb, channels="RGB")
    zone1_count.metric("Camera Count", count1)
    zone1_density.metric("Density", density1)
    zone1_chart.line_chart(pd.DataFrame(zone1_graph))

    zone2_video.image(frame2_rgb, channels="RGB")
    zone2_count.metric("Camera Count", count2)
    zone2_density.metric("Density", density2)
    zone2_chart.line_chart(pd.DataFrame(zone2_graph))

    total_placeholder.header(
        f"👥 Total People Inside Area: {total_people}"
    )

    if overcrowded:
        alert_placeholder.error("🚨 OVERCROWDED")
    else:
        alert_placeholder.success("✅ SAFE")

    time.sleep(0.03)