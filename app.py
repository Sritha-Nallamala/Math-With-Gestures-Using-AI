import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import streamlit as st
import google.generativeai as genai

st.set_page_config(layout="wide")
st.title("AI-Powered Hand-Drawn Recognizer")

col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox("Run", value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.subheader("AI Recognition Result")
    output_text_area = st.empty()

# ---------------- AI Setup ----------------
genai.configure(api_key="YOUR_API_KEY") 
model = genai.GenerativeModel('gemini-2.5-flash')

# ---------------- Webcam & Hand Detector ----------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(maxHands=1, detectionCon=0.7, minTrackCon=0.5)

# ---------------- Helper Functions ----------------
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]: 
        current_pos = tuple(lmList[8][0:2])
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, prev_pos, current_pos, (255, 0, 255), 10)
    elif fingers == [1, 0, 0, 0, 0]: 
        canvas = np.zeros_like(canvas)
    return current_pos, canvas

def sendToAI(canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:
        pil_image = Image.fromarray(canvas)
        prompt = "Identify what is drawn in this image. If it's a math expression, calculate the result. If it's a shape or diagram, describe it."
        response = model.generate_content([prompt, pil_image])
        return response.text
    return ""

prev_pos = None
canvas = None
output_text = ""

while run:
    success, img = cap.read()
    if not success:
        continue
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(canvas, fingers)

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

    if output_text:
        output_text_area.text(output_text)

cap.release()
cv2.destroyAllWindows()
