import streamlit as st
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from google.cloud import storage
import requests


# Experimenting banner creation
BANNERBEAR_API_KEY = st.secrets["BANNERBEAR_API_KEY"]
BANNERBEAR_TEMPLATE = st.secrets["BANNERBEAR_TEMPLATE_UID"]


def create_banner(path, timestamp, style):
    # Experimenting with banner bear api
    base, ext = os.path.splitext(path)
    bucket_name = st.secrets["GCS_BUCKET"]
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(os.path.basename(path))
    blob.upload_from_filename(path)
    video_uri = f"gs://{bucket_name}/{os.path.basename(path)}"

    # Format timestamp seconds into HH:MM:SS
    h = int(timestamp // 3600)
    m = int((timestamp % 3600) // 60)
    s = int(timestamp % 60)
    ts = f"{h:02d}:{m:02d}:{s:02d}"

    url = "https://api.bannerbear.com/v2/video_frames"
    payload = {
        "video_url": video_uri,
        "timestamp": ts,
        "template": BANNERBEAR_TEMPLATE,
        "modifications": [
            {"name": "title", "text": st.session_state.video_title},
            {"name": "subtitle", "text": style}
        ]
    }
    headers = {"Authorization": f"Bearer {BANNERBEAR_API_KEY}", "Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers)
    data = resp.json()

    return data.get("image_url"), data.get("self")


def extract_clip(path, start, end, outp):
    return ffmpeg_extract_subclip(path, start, end, outp)
