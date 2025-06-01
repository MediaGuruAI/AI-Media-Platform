import tempfile, os, shutil, time, subprocess, base64, requests
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy import AudioFileClip, VideoFileClip, concatenate_videoclips
import google.generativeai as genai
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import videointelligence_v1 as vi, vision
import boto3
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import json
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from pathlib import Path
from google.oauth2 import service_account

def is_av1_encoded(video_path):
    """Check if a video is AV1-encoded using FFmpeg."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",  # Check first video stream
        "-show_entries", "stream=codec_name",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout.strip().lower() == "av1"
    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg: {e.stderr}")
        return False
    

def convert_av1_to_h264(input_path, output_path=None):
    """More reliable conversion with progress tracking"""
    
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_converted.mp4"
    
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite without asking
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "128k",
        "-max_muxing_queue_size", "1024",  # Prevents muxer errors
        output_path
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait for completion with timeout (10 minutes)
        _, stderr = process.communicate(timeout=600)
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg error: {stderr[:500]}...")            
        return output_path
    
    except subprocess.TimeoutExpired:
        process.kill()
        raise Exception("Conversion timed out after 10 minutes")
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise Exception(f"Conversion failed: {str(e)}") from e


# Defining Helper Functions
# Function for extracting video duration 
def get_video_duration(path):
    clip = VideoFileClip(path)
    duration = clip.duration
    clip.close()
    return duration

# Saving video in temp dir
def save_temp_video(uploaded):
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.read())
    return path, temp_dir


# Function for transcribing the video in chunks (spiliting video under 60sec)
def transcribe(path, credentials, chunk_length=59, overlap=0.5, save_dir=None):
    """
    Splits audio into overlapping chunks, transcribes each in multiple languages,
    and returns both full text and per-chunk transcripts with timestamps.
    """
    base, _ = os.path.splitext(path)
    print(path)
    clip = AudioFileClip(path)
    duration = clip.duration
    transcripts = []
    transcripts_dir = f"{save_dir}/transcripts"
    Path(transcripts_dir).mkdir(parents=True)
    credentials = service_account.Credentials.from_service_account_file(credentials["google_credentials"])
    speech_client = speech.SpeechClient(credentials=credentials)

    def _transcribe_segment(start, end):
        fn = f"{base}_chunk_{int(start*1000)}.wav"
        clip.subclipped(start, end).write_audiofile(
            fn, fps=16000, ffmpeg_params=["-ac", "1"], logger=None
        )
        with open(fn, "rb") as f:
            audio = speech.RecognitionAudio(content=f.read())
        os.remove(fn)

        best = ""
        for lang in ("en-US", "ar", "ur"):
            cfg = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=lang,
                enable_automatic_punctuation=True
            )
            try:
                resp = speech_client.recognize(config=cfg, audio=audio)
                text = " ".join(r.alternatives[0].transcript for r in resp.results)
                if len(text) > len(best):
                    best = text
            except Exception:
                continue
        return best.strip()

    if duration <= chunk_length:
        text = _transcribe_segment(0, duration)
        transcripts.append({"start": 0, "end": duration, "text": text})
    else:
        start = 0.0
        while start < duration:
            end = min(start + chunk_length, duration)
            text = _transcribe_segment(start, end)
            transcripts.append({"start": start, "end": end, "text": text})
            start += chunk_length - overlap
            time.sleep(0.2)

    # filter out empty chunks
    segments = [s for s in transcripts if s["text"]]
    full_text = " ".join(s["text"] for s in segments)
    save_transcript(path, full_text, segments, transcripts_dir)
    return transcripts_dir, full_text, segments


# saving and loading transcript 
# defining trancript path 
def transcript_paths(video_path, transcript_dir):
    """Return the JSON path you’ll use to save/load this video’s transcript."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(transcript_dir, f"{base}.json")

# saving trancript 
def save_transcript(video_path, full, segments, transcript_dir):
    """Persist the full text and per-chunk segments as JSON."""
    p = transcript_paths(video_path, transcript_dir)
    with open(p, "w") as f:
        json.dump({"full": full, "segments": segments}, f, indent=2)

def save_scenes(video_path, scenes, SCENE_DIR):
    p = scene_path(video_path, SCENE_DIR)
    with open(p, "w") as f:
        json.dump(scenes, f, indent=2)

def scene_path(video_path, SCENE_DIR):
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(SCENE_DIR, f"{base}.json")

# Breaking vide in frames and analysing frames
def analyze_frame(image_bytes, credentials):
    
    rekognition = boto3.client(
        'rekognition',
        aws_access_key_id=credentials["aws_access_key"],
        aws_secret_access_key=credentials["aws_secret_key"],
        region_name="us-east-1"
    )
    google_credentials = service_account.Credentials.from_service_account_file(credentials["google_credentials"])   
    vision_client = vision.ImageAnnotatorClient(credentials=google_credentials)     
    labels = []
    try:
        # AWS Rekognition
        aws_labels = rekognition.detect_labels(Image={"Bytes": image_bytes}, MaxLabels=15)
        labels += [l["Name"] for l in aws_labels.get("Labels", [])]
        
        # Celebrity Recognition
        celebs = rekognition.recognize_celebrities(Image={"Bytes": image_bytes})
        labels += [c["Name"] for c in celebs.get("CelebrityFaces", [])]
        
        # Google Vision
        gimg = vision.Image(content=image_bytes)
        gv_labels = vision_client.label_detection(image=gimg)
        labels += [l.description for l in gv_labels.label_annotations]
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")
    
    return list(dict.fromkeys(labels))[:20]


# For scene detection and  generating summaries of scenes 
def enrich_shots(path, transcript_segments, save_dir, credentials):
    """
    Detects shot changes, analyzes frames, and generates LLM summaries including
    both labels and the transcript for that scene.
    """
    google_credentials = service_account.Credentials.from_service_account_file(credentials["google_credentials"])
    vi_client = vi.VideoIntelligenceServiceClient(credentials=google_credentials)
    client = OpenAI(api_key=credentials["openai_key"])

    # check if video is av1_encoded
    if is_av1_encoded(path):
        path = convert_av1_to_h264(path)

    with open(path, "rb") as f:
        content = f.read()

    req = vi.AnnotateVideoRequest(
        input_content=content,
        features=[vi.Feature.SHOT_CHANGE_DETECTION]
    )
    op = vi_client.annotate_video(request=req)
    shots = op.result(timeout=300).annotation_results[0].shot_annotations
    scenes_dir = f"{save_dir}/scenes"
    Path(scenes_dir).mkdir(parents=True)
    records = []
    for idx1, shot in enumerate(shots):
        start = shot.start_time_offset.total_seconds()
        end = shot.end_time_offset.total_seconds()
        labels = set()
        # analyze frames
        for idx2, t in enumerate([start + (end - start) * i / 3 for i in (0, 1, 2)]):
            frame_path = save_frame(path, t, f"{scenes_dir}/frame_{idx1 + idx2}.jpg")
            with open(frame_path, "rb") as f:
                labels.update(analyze_frame(f.read(), credentials))
            # os.remove(frame_path)

        # extract transcript for this scene
        scene_text = " ".join(
            seg["text"] for seg in transcript_segments
            if seg["start"] < end and seg["end"] > start
        )

        # Generate scene summary with transcript context
        try:
            resp = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": (
                        
                        """You are a professional video‐scene summarization engine. For each shot, generate a clear, 120–150-word paragraph that:

                        Names any people, characters, or figures you can identify (and their roles, if known).

                        Describes the key visual elements and setting.

                        Summarizes any actions or interactions.

                        Captures the emotional tone or atmosphere.

                        References any spoken text when relevant.

                        Write in a neutral, reportorial style—concise but vivid. Avoid filler or speculation beyond what’s visible."""
                    )},
                    {"role": "user", "content": (
                        f"Shot start–end: {start}s–{end}s\n"
                        f"Detected elements: {', '.join(labels)}\n"
                        f"Transcript excerpt: {scene_text if scene_text else '[No speech]'}"
                    )}
                ],
                temperature=0.7,
                # max_tokens=150
                max_tokens=300
            )
            summary = resp.choices[0].message.content.strip()
        except Exception as e:
            summary = "Scene analysis unavailable"
            print(f"GPT-4 error: {str(e)}")

        records.append({
            "start": start,
            "end": end,
            "labels": list(labels),
            "summary": summary
        })
    save_scenes(path, records, scenes_dir)
    return records

# Saving each frame
def save_frame(path, t, outp):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(outp, frame)
    cap.release()
    return outp

# Indexing scenes on pinecone
def index_scenes(scenes, credentials):
    vectors = []
    index = None
    client = OpenAI(api_key=credentials["openai_key"])
    for i, scene in enumerate(scenes):
        text = f"{scene['summary']} {' '.join(scene['labels'])}"
        emb = client.embeddings.create(input=[text], model="text-embedding-ada-002")
        vectors.append((str(i), emb.data[0].embedding, scene))
    index.upsert(vectors=vectors)
    return index

# for searching the best match for prompt
def semantic_search(prompt, credentials, index, top_k=3):

    client = OpenAI(api_key=credentials["openai_key"])
    emb = client.embeddings.create(input=[prompt], model="text-embedding-ada-002")
    return index.query(
        vector=emb.data[0].embedding,
        top_k=top_k,
        include_metadata=True
    )

# generating Scene summaries in chunks
def summarize_chunks_with_gpt4(summaries, credentials, chunk_size=5):

    client = OpenAI(api_key=credentials["openai_key"])
    chunk_summaries = []

    system_msg = (
        "You are an expert video summarization assistant. Your task is to summarize a group of related video scenes "
        "into a single, fluent paragraph of 80–100 words. Capture:\n"
        "- The main idea or purpose of the scenes\n"
        "- Key events and actions\n"
        "- Important people or characters\n"
        "- Mentioned locations (if any)\n"
        "- Emotional tone or insights when relevant\n"
        "The summary should feel natural, informative, and capture the essence of what occurred in the scenes."
    )

    for i in range(0, len(summaries), chunk_size):
        chunk = summaries[i:i + chunk_size]
        user_msg = "\n".join(f"- {s}" for s in chunk)

        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=800,
            temperature=0.7
        )

        summary = resp.choices[0].message.content.strip()
        chunk_summaries.append(summary)
        time.sleep(1)  # helps avoid hitting rate limits

    return chunk_summaries


# Generating final full summary 
def final_summary_with_openai(chunk_summaries, full_text, credentials, video_path, save_dir):

    client = OpenAI(api_key=credentials["openai_key"])
    system_msg = "You are a professional video summarization expert. "
    user_msg = (

        "Combine these chunk summaries into a cohesive and comprehensive summary of the entire video in 150–200 words. "
        "Focus on the core idea, key events, people involved, locations, and any significant developments. "
        "Ensure the summary flows logically, highlights the central message, and captures the essence of the video without unnecessary detail.\n\n"
        + "\n".join(f"- {s}" for s in chunk_summaries)
        + "\n\nTranscript Excerpt (for context if needed):\n"
        + (full_text[:5000] + "…")

    )
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=800,
        temperature=0.7
    )
    summary_text = resp.choices[0].message.content.strip()
    
    summary_dir = f"{save_dir}/summary"
    Path(summary_dir).mkdir(parents=True)
    save_summary(video_path, summary_text, summary_dir)
    return summary_text

# loading and saving full summary
def summary_path(video_path, SUMMARY_DIR):
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(SUMMARY_DIR, f"{base}.txt")

# def load_summary(video_path):
#     p = summary_path(video_path)
#     return open(p).read() if os.path.exists(p) else None

def save_summary(video_path, text, SUMMARY_DIR):
    with open(summary_path(video_path, SUMMARY_DIR), "w") as f:
        f.write(text)


# for extracting the main agenda for banner title
def extract_key_line(transcript: str, summary: str, credentials: dict) -> str:

    """
    Ask the LLM to return the single most important line (agenda) in the video,
    by looking at both the full transcript and the full summary.
    Output must be exactly one line.
    """
    client = OpenAI(api_key=credentials["openai_key"])
    combined = (
        "Transcript:\n" + transcript + "\n\n" +
        "Summary:\n" + summary
    )
    prompt = (
        "Given the full video transcript and a concise summary, identify the single most important headline or agenda line that captures the video's core message. "
        "Output exactly one line.\n\n" + combined
    )
    resp = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a video summarization expert that provides a single headline line."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=50
    )
    # ensure single line
    return resp.choices[0].message.content.strip().splitlines()[0]


# for searching the best match for prompt
def semantic_search(prompt, index, credentials, top_k=3):

    client = OpenAI(api_key=credentials["openai_key"])
    emb = client.embeddings.create(input=[prompt], model="text-embedding-ada-002")
    return index.query(
        vector=emb.data[0].embedding,
        top_k=top_k,
        include_metadata=True
    )

# function for better scene index management
def index_scenes(scenes, index, credentials):

    # delete any old vectors so IDs always line up 0…len(scenes)-1
    index.delete(delete_all=True)
    vectors = []
    client = OpenAI(api_key=credentials["openai_key"])
    for i, scene in enumerate(scenes):
        text = f"{scene['summary']} {' '.join(scene['labels'])}"
        emb = client.embeddings.create(input=[text], model="text-embedding-ada-002")
        vectors.append((str(i), emb.data[0].embedding, scene))
    index.upsert(vectors=vectors)
    return index

# Image processing utilities
def image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# validating image path
def is_valid_image(file_path):
    try:
        Image.open(file_path)
        return True
    except Exception:
        return False

# Resizing image
def resize_image(input_path, output_path, size=(846, 541)):
    with Image.open(input_path) as img:
        img = img.resize(size)
        img.save(output_path)

# Save and load banners
def banner_path(video_path, save_dir):
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(save_dir, f"{base}.png")

def load_banner(video_path):
    p = banner_path(video_path)
    return p if os.path.exists(p) else None

def save_banner(video_path, img_bytes, save_dir):
    p = banner_path(video_path, save_dir)
    with open(p, "wb") as f:
        f.write(img_bytes)


