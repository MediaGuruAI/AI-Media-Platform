import streamlit as st
import time
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy import AudioFileClip
from google.cloud import speech_v1p1beta1 as speech




# Defining trancribe function 
def transcribe(path):
    st.info("Transcribing with Google Cloud Speech‑to‑Text…")
    
    # building output path
    base, _ = os.path.splitext(path)
    audio_path = base + "_audio.wav"
    
    clip = AudioFileClip(path)
    

    clip.write_audiofile(
        audio_path,
        fps=16000,
        ffmpeg_params=["-ac", "1"],   
        logger=None
    )
    

    with open(audio_path, "rb") as f:
        audio_content = f.read()
    os.remove(audio_path)

    # configuring Google Speech API
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True
    )
    response = speech.SpeechClient().recognize(config=config, audio=audio)
    
    # returning joined transcript
    time.sleep(5)
    return " ".join(r.alternatives[0].transcript for r in response.results)
