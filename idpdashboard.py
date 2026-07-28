import streamlit as st
import cv2
import pandas as pd
import time
import serial
from ultralytics import YOLO

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI Crowd Monitor", layout="wide")

# ===============================
# LOAD MODEL
# ===============================
model = YOLO("yolov8n.pt")

# ===============================
# SERIAL (CHANGE PORT)
# ===============================
ser = serial.Serial('COM16', 115200)

# ===============================
# CAMERA
# ===============================
cap = cv2.VideoCapture(0)

zone_graph = []
frame_count = 0
ir_count = 0

# ===============================
# PROCESS FUNCTION
# ===============================
def process_zone(frame):

    frame = cv2.resize(frame, (700, 400))
    height, width = frame.shape[:2]
    frame_area = width * height

    results = model(frame)[0]

    people = 0
    area = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == 0 and conf > 0.3:
            people += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            area += w * h

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    density_ratio = area / frame_area

    if density_ratio > 0.25:
        density = "HIGH"
    elif density_ratio > 0.12:
        density = "MEDIUM"
    else:
        density = "LOW"

    return frame, people, density

# ===============================
# UI
# ===============================
zone_video = st.empty()
zone_count = st.empty()
zone_density = st.empty()
zone_chart = st.empty()

st.markdown("---")
total_placeholder = st.empty()
alert_placeholder = st.empty()

# ===============================
# LOOP
# ===============================
while True:

    frame_count += 1

    # ===== READ IR COUNT =====
    if ser.in_waiting:
        try:
            ir_count = int(ser.readline().decode().strip())
        except:
            pass

    ret, frame = cap.read()

    if not ret:
        st.error("Camera not working")
        break

    if frame_count % 2 != 0:
        continue

    frame, count, density = process_zone(frame)

    total_people = ir_count

    # ==========================
    # ✅ UPDATED OVERCROWD LOGIC
    # ==========================
    overcrowded = False

    # 🔹 Condition 1: Total capacity (IR sensors)
    if total_people > 20:
        overcrowded = True

    # 🔹 Condition 2: Camera zone (MAIN FIX)
    if count > 5:
        overcrowded = True

    # ==========================
    # SEND TO ESP32
    # ==========================
    if overcrowded:
        ser.write(b'1')
    else:
        ser.write(b'0')

    # ==========================
    # GRAPH
    # ==========================
    if frame_count % 20 == 0:
        zone_graph.append(count)
        if len(zone_graph) > 20:
            zone_graph.pop(0)

    # ==========================
    # DISPLAY
    # ==========================
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    zone_video.image(frame_rgb, channels="RGB")
    zone_count.metric("Live Camera Count (Zone)", count)
    zone_density.metric("Density", density)
    zone_chart.line_chart(pd.DataFrame(zone_graph))

    total_placeholder.header(f"👥 Total People (IR Sensors): {total_people}")

    if overcrowded:
        alert_placeholder.error("🚨 OVERCROWDED")
    else:
        alert_placeholder.success("✅ SAFE")

    time.sleep(0.03)