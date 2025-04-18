import streamlit as st
import tempfile, os, shutil, time
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy import AudioFileClip
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import videointelligence_v1 as vi, vision
import boto3
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
import google.generativeai as genai
from google.cloud import storage
import requests


# Settingup GCP & AWS clients for rekognition
vi_client = vi.VideoIntelligenceServiceClient()
vision_client = vision.ImageAnnotatorClient()
rekognition = boto3.client(
    'rekognition',
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY"],
    aws_secret_access_key=st.secrets["AWS_SECRET_KEY"],
    region_name="us-east-1"
)



# Defining code for scenes detection 
def detect_scenes(path):
    st.info("Detecting shots…")
    with open(path, "rb") as f: content = f.read()
    req = vi.AnnotateVideoRequest(
        input_content=content,
        features=[vi.Feature.SHOT_CHANGE_DETECTION]
    )
    op = vi_client.annotate_video(request=req)
    time.sleep(5)  # Addeding sleep 
    return op.result(timeout=300).annotation_results[0].shot_annotations

def save_frame(path, t, outp):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ret, frame = cap.read()
    if ret: cv2.imwrite(outp, frame)
    cap.release()
    return outp

def analyze_frame(image_bytes):
    labels = []
    try:
        aws = rekognition.detect_labels(Image={"Bytes": image_bytes}, MaxLabels=10)
        labels += [l["Name"] for l in aws.get("Labels", [])]
    except Exception as e: st.error(e)
    try:
        gimg = vision.Image(content=image_bytes)
        gv = vision_client.label_detection(image=gimg)
        labels += [l.description for l in gv.label_annotations]
    except Exception as e: st.error(e)
    time.sleep(5)  # Addeding sleep
    return list(dict.fromkeys(labels))[:15]

def summarize_scene(transcript, labels):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = (
        f"Create a concise scene summary (2–3 sentences) capturing:\n"
        f"- Visual elements: {', '.join(labels)}\n"
        f"- Transcript context: {transcript}\n"
        f"- Emotional tone"
    )
    return model.generate_content(prompt).text.strip()
