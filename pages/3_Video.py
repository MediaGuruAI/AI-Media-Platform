import streamlit as st
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

# Page Config - Set initial sidebar state to collapsed
st.set_page_config(
    page_title="Audio Processor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme matching main page
st.markdown("""
<style>
    /* Main page styling */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(11deg, #0c292070, #717054, #24243e) !important;
        color: white !important;
    }
    
    /* Hide the default sidebar navigation */
    [data-testid="stSidebarNav"] {
        display: none;
    }
    
    /* Title styling */
    .title {
        font-size: 5rem !important;
        font-weight: 800;
        background: linear-gradient(90deg, #4ecdc4, #88f3e8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Card styling */
    .card {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 20px;
    }
    
    /* Button styling */
    .stButton>button {
        border: 2px solid #4ecdc4 !important;
        background: transparent !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        background: #4ecdc4 !important;
    }
    
    /* File uploader styling */
    .stFileUploader>div>div>div>div {
        color: white !important;
    }
    
    /* JSON viewer styling */
    .stJson {
        background: rgba(0, 0, 0, 0.2) !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
            
        [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.9) !important;
        border-right: 1px solid rgba(255, 107, 107, 0.2) !important;
    }

    [data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem !important;
    }

    [data-testid="stSidebar"] h2 {
        color: #ff6b6b !important;
    }

    [data-testid="stSidebar"] a {
        color: rgba(255, 255, 255, 0.8) !important;
        text-decoration: none !important;
        display: block;
        padding: 0.5rem 0;
    }

    [data-testid="stMarkdownContainer"] p{
            color: white;        
    }
    [data-testid="stSidebar"] a:hover {
        color: #ff6b6b !important;
    }

    .sidebar-toggle {
        position: fixed;
        left: 10px;
        top: 10px;
        z-index: 999999;
        background: rgba(15, 12, 41, 0.9) !important;
        border: 1px solid #ff6b6b !important;
        color: white !important;
        padding: 5px 10px !important;
        border-radius: 5px !important;
    }
            
    .sidebar-toggle:hover {
        background: #ff6b6b !important;
    }
</style>
""", unsafe_allow_html=True)


# Configuration & Clients Setup
OFFLINE = os.getenv("OFFLINE_MODE", "0") == "1"

# Google Cloud Setup (only if online)
if not OFFLINE:
    google_credentials = st.secrets.get("google_cloud", {})
    credentials_dict = {
        "type": "service_account",
        "project_id": google_credentials.get("project_id", ""),
        "private_key_id": google_credentials.get("private_key_id", ""),
        "private_key": google_credentials.get("private_key", "").replace('\\n', '\n'),
        "client_email": google_credentials.get("client_email", ""),
        "client_id": google_credentials.get("client_id", ""),
        "auth_uri": google_credentials.get("auth_uri", ""),
        "token_uri": google_credentials.get("token_uri", ""),
        "auth_provider_x509_cert_url": google_credentials.get("auth_provider_x509_cert_url", ""),
        "client_x509_cert_url": google_credentials.get("client_x509_cert_url", "")
    }
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as temp_file:
        json.dump(credentials_dict, temp_file)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file.name

# Initialize external clients with offline fallback
client = None
speech_client = None
vi_client = None
vision_client = None
rekognition = None
index = None

if not OFFLINE:
    try:
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))
    except Exception:
        client = None

    try:
        genai.configure(api_key=st.secrets.get("GOOGLE_API_KEY", ""))
    except Exception:
        pass

    try:
        speech_client = speech.SpeechClient()
        vi_client = vi.VideoIntelligenceServiceClient()
        vision_client = vision.ImageAnnotatorClient()
    except Exception:
        speech_client = vi_client = vision_client = None

    try:
        rekognition = boto3.client(
            'rekognition',
            aws_access_key_id=st.secrets.get("AWS_ACCESS_KEY", ""),
            aws_secret_access_key=st.secrets.get("AWS_SECRET_KEY", ""),
            region_name="us-east-1"
        )
    except Exception:
        rekognition = None

    # Pinecone
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))
        if "video-highlights" not in pc.list_indexes().names():
            pc.create_index(
                name="video-highlights",
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        index = pc.Index("video-highlights")
    except Exception:
        index = None


# Setting up Templated.io 
TEMPLATED_API_URL = "https://api.templated.io/v1/renders"
TEMPLATED_API_KEY = st.secrets["TEMPLATED_API_KEY"]
TEMPLATED_TEMPLATE_ID = st.secrets["TEMPLATED_TEMPLATE_ID"]

# Making local dir
# Transcript
TRANSCRIPT_DIR = "transcripts"
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

#Scenes 
SCENE_DIR = "scenes"
os.makedirs(SCENE_DIR, exist_ok=True)

# full summary 
SUMMARY_DIR = "full_summaries"
os.makedirs(SUMMARY_DIR, exist_ok=True)

# banner
BANNER_DIR = "banner"
os.makedirs(BANNER_DIR, exist_ok=True)

# Promo
PROMO_DIR = "promo"
os.makedirs(PROMO_DIR, exist_ok=True)


#Webapp title
st.title("🎬 AI Video Analyzer")

# Back button
if st.button("← Back to Main Page"):
    st.switch_page("app.py")


# Session defaults
st.session_state.setdefault("uploaded", None)
st.session_state.setdefault("transcript", "")
st.session_state.setdefault("segments", [])
st.session_state.setdefault("scenes", [])
st.session_state.setdefault("scene_records", [])
st.session_state.setdefault("selected_scenes", [])
st.session_state.setdefault("banner_path", None)
st.session_state.setdefault("promo_path", None)
st.session_state.setdefault("full_summary", None)
st.session_state.setdefault("chunk_transcripts", None)

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
def transcribe(path, chunk_length=59, overlap=0.5):
    """
    Splits audio into overlapping chunks, transcribes each in multiple languages,
    and returns both full text and per-chunk transcripts with timestamps.
    """
    base, _ = os.path.splitext(path)
    clip = AudioFileClip(path)
    duration = clip.duration
    transcripts = []

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
    return full_text, segments


# saving and loading transcript 
# defining trancript path 
def transcript_paths(video_path):
    """Return the JSON path you’ll use to save/load this video’s transcript."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(TRANSCRIPT_DIR, f"{base}.json")

# loading transcript
def load_transcript(video_path):
    """If we’ve already saved a transcript for this video, load and return it."""
    p = transcript_paths(video_path)
    if os.path.exists(p):
        with open(p, "r") as f:
            data = json.load(f)
        return data["full"], data["segments"]
    return None, None


# saving trancript 
def save_transcript(video_path, full, segments):
    """Persist the full text and per-chunk segments as JSON."""
    p = transcript_paths(video_path)
    with open(p, "w") as f:
        json.dump({"full": full, "segments": segments}, f, indent=2)


# Breaking vide in frames and analysing frames
def analyze_frame(image_bytes):
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
        st.error(f"Analysis error: {str(e)}")
    
    return list(dict.fromkeys(labels))[:20]

# For scene detection and  generating summaries of scenes 
def enrich_shots(path, transcript_segments):
    """
    Detects shot changes, analyzes frames, and generates LLM summaries including
    both labels and the transcript for that scene.
    """

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

    records = []
    for shot in shots:
        start = shot.start_time_offset.total_seconds()
        end = shot.end_time_offset.total_seconds()
        labels = set()
        # analyze frames
        for t in [start + (end - start) * i / 3 for i in (0, 1, 2)]:
            frame_path = save_frame(path, t, f"frame_{t}.jpg")
            with open(frame_path, "rb") as f:
                labels.update(analyze_frame(f.read()))
            os.remove(frame_path)

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
            st.error(f"GPT-4 error: {str(e)}")

        records.append({
            "start": start,
            "end": end,
            "labels": list(labels),
            "summary": summary
        })

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
def index_scenes(scenes):
    vectors = []
    for i, scene in enumerate(scenes):
        text = f"{scene['summary']} {' '.join(scene['labels'])}"
        emb = client.embeddings.create(input=[text], model="text-embedding-ada-002")
        vectors.append((str(i), emb.data[0].embedding, scene))
    index.upsert(vectors=vectors)


# loading and saving scenes
def scene_path(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(SCENE_DIR, f"{base}.json")

def load_scenes(video_path):
    p = scene_path(video_path)
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return None

def save_scenes(video_path, scenes):
    p = scene_path(video_path)
    with open(p, "w") as f:
        json.dump(scenes, f, indent=2)


# generating Scene summaries in chunks
def summarize_chunks_with_gpt4(client, summaries, chunk_size=5):
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
def final_summary_with_openai(chunk_summaries , full_text):
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
    return resp.choices[0].message.content.strip()

# loading and saving full summary
def summary_path(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(SUMMARY_DIR, f"{base}.txt")

def load_summary(video_path):
    p = summary_path(video_path)
    return open(p).read() if os.path.exists(p) else None

def save_summary(video_path, text):
    with open(summary_path(video_path), "w") as f:
        f.write(text)


# for extracting the main agenda for banner title
def extract_key_line(transcript: str, summary: str, client) -> str:
    """
    Ask the LLM to return the single most important line (agenda) in the video,
    by looking at both the full transcript and the full summary.
    Output must be exactly one line.
    """
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
def semantic_search(prompt, top_k=3):
    emb = client.embeddings.create(input=[prompt], model="text-embedding-ada-002")
    return index.query(
        vector=emb.data[0].embedding,
        top_k=top_k,
        include_metadata=True
    )

# function for better scene index management
def index_scenes(scenes):
    # delete any old vectors so IDs always line up 0…len(scenes)-1
    index.delete(delete_all=True)
    vectors = []
    for i, scene in enumerate(scenes):
        text = f"{scene['summary']} {' '.join(scene['labels'])}"
        emb = client.embeddings.create(input=[text], model="text-embedding-ada-002")
        vectors.append((str(i), emb.data[0].embedding, scene))
    index.upsert(vectors=vectors)

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


# Saving and loading promo clip
def promo_path(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(PROMO_DIR, f"{base}_promo.mp4")

def load_promo(video_path):
    p = promo_path(video_path)
    return p if os.path.exists(p) else None


# Save and load banners
def banner_path(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(BANNER_DIR, f"{base}.png")

def load_banner(video_path):
    p = banner_path(video_path)
    return p if os.path.exists(p) else None

def save_banner(video_path, img_bytes):
    p = banner_path(video_path)
    with open(p, "wb") as f:
        f.write(img_bytes)



#  Configuration for thumbnail
DEFAULT_FONT = "arial.ttf"  
TITLE_FONT_SIZE = 60
SUBTITLE_FONT_SIZE = 40
LOGO_SIZE = (80, 80)
PADDING = 30
BAR_HEIGHT = 120
BAR_COLOR = (0, 123, 255, 200)  
TEXT_COLOR = (255, 255, 255, 255)
SHADOW_COLOR = (0, 0, 0, 150)



if 'scene_records' not in st.session_state:
    st.session_state['scene_records'] = []

uploaded = st.file_uploader("Upload video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])
if uploaded:
    video_path, temp_dir = save_temp_video(uploaded)
    duration = get_video_duration(video_path)
    st.session_state.uploaded = video_path
    

    if st.button("1️⃣ Transcribe Video", type="primary"):
        full, segments = load_transcript(video_path)
        if full is not None:
            st.success("Loaded transcript from cache.")
        else:
            # full, segments = transcribe(st.session_state.uploaded)
            full, segments = transcribe(video_path)
            save_transcript(video_path, full, segments)
            st.success("Transcription complete and saved.")
        st.session_state.transcript = full
        st.session_state.chunk_transcripts = segments



    if duration > 240:  # 4 minutes
        st.warning("Note: Full analysis limited to first 4 minutes for free tier services")

    # Processing Buttons 
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✨ Analyze Scenes", type="primary"):
            with st.spinner():
                # try cache first
                cached = load_scenes(video_path)
                if cached is not None:
                    st.session_state.scenes = cached
                    st.session_state.scene_records = cached
                    st.success(f"Loaded {len(cached)} scenes from cache.")
                else:
                    shots = enrich_shots(video_path, st.session_state.chunk_transcripts)
                    st.session_state.scenes = shots
                    st.session_state.scene_records = shots
                    save_scenes(video_path, shots)
                    st.success(f"Analyzed and saved {len(shots)} scenes.")
        

    # Defining tab setup
    tab0, tab1, tab6, tab2, tab5, tab3, tab4,  = st.tabs([
        "Transcription" , "Scene Breakdown", "Full video summary", "Semantic Search", 
        "Promo Clip", "Banner Creator", "Download data" 
    ])

    st.session_state.setdefault("selected_scenes", [])
 
    with tab0:
        st.header("Full Transcript")
       
        if st.session_state.get('transcript'):
            st.text_area(
                "Combined Transcript",
                value=st.session_state['transcript'],
                height=300
            )
        else:
            st.info("Click **1️⃣ Transcribe Video** first.")


    # tab1 for scene summaries 
    with tab1:  # Scene Breakdown
        if st.session_state.scenes:
            st.subheader("📽️ Scene Analysis")
            for i, scene in enumerate(st.session_state.scenes):
                with st.expander(f"Scene {i+1} ({scene['start']:.1f}s - {scene['end']:.1f}s)"):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        frame_path = save_frame(video_path, scene["start"], f"scene_{i}.jpg")
                        st.image(frame_path, width=200)
                        
                        st.write(f"**Detected Elements:** {', '.join(scene['labels'])}")
                    with col2:
                        st.write(f"**AI Summary:** {scene['summary']}")
        else:
            st.info("Click 'Analyze Scenes' to process video content")


    with tab2:
        st.subheader("🔍 Find Moments by Description")
        query = st.text_input("Describe the moment you want to find:", 
                            placeholder="e.g., 'emotional speech', 'action sequence', 'beautiful landscape'")
        
        if query and st.session_state.scenes:
            results = semantic_search(query)
            st.subheader("Top Matching Scenes")
            
            # Reset selection when new search
            if query != st.session_state.get("last_query", ""):
                st.session_state.selected_scenes = []
            st.session_state.last_query = query
            
            for i, match in enumerate(results["matches"]):
                scene_idx = int(match["id"])
                if 0 <= scene_idx < len(st.session_state.scenes):
                    scene = st.session_state.scenes[scene_idx]
                else:
                    continue 

                scene = st.session_state.scenes[scene_idx]
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    frame_path = save_frame(video_path, scene["start"], f"match_{i}.jpg")
                    st.image(frame_path, caption=f"{scene['end']-scene['start']:.1f}s scene")
                with col2:
                    # Add checkbox
                    selected = st.checkbox(
                        f"Select this scene (Score: {match['score']:.2f})",
                        value=scene_idx in st.session_state.selected_scenes,
                        key=f"select_{i}"
                    )
                    
                    if selected and scene_idx not in st.session_state.selected_scenes:
                        st.session_state.selected_scenes.append(scene_idx)
                    elif not selected and scene_idx in st.session_state.selected_scenes:
                        st.session_state.selected_scenes.remove(scene_idx)
                    
                    st.write(f"**Time:** {scene['start']:.1f}s - {scene['end']:.1f}s")
                    st.write(f"**Summary:** {scene['summary']}")
                    st.write(f"**Labels:** {', '.join(scene['labels'][:10])}")



    with tab3:
        if st.button("🤖 Generate Modern Banner", type="primary"):
            # Preconditions: scenes + text
            scenes = st.session_state.get('scenes', [])
            if not scenes:
                st.error("⚠️ Run Scene Analysis first!")
                st.stop()

            # ensure transcript & full summary
            transcript = st.session_state.get('transcript', '')
            full_sum = st.session_state.get('full_summary')
            if not full_sum:
                sums = [r['summary'] for r in st.session_state.get('scene_records', [])]
                full_sum = final_summary_with_openai(sums, transcript)
                # save_summary(video_path, full_sum)
                st.session_state['full_summary'] = full_sum

            # generate banner text
            banner_title = extract_key_line(transcript, full_sum, client)
            st.write(banner_title)
            title_font_size = str(max(24, 60 - len(banner_title)//3)) + "px"

            # pick a representative frame
            res = semantic_search(banner_title, top_k=3)
            valid = [m for m in res.get('matches', []) if m.get('id', '').isdigit()]
            if not valid:
                st.error("No valid scene—re-run analysis.")
                st.stop()
            scene = scenes[int(valid[0]['id'])]

            # prepare background image
            frame_path = save_frame(video_path, scene['start'], 'auto_frame.jpg')
            resized_path = 'resized_frame.jpg'
            resize_image(frame_path, resized_path, size=(846, 541))
            if not (is_valid_image(resized_path) and os.path.getsize(resized_path) <= 5 * 1024 * 1024):
                st.error("Invalid or too-large resized frame image")
                st.stop()
            bg_data = image_to_base64(resized_path)

            # call Templated API
            api_key = st.secrets['TEMPLATED_API_KEY']
            template_id = st.secrets['TEMPLATED_TEMPLATE_ID']
            url = 'https://api.templated.io/v1/render'
            headers = {'Authorization': f'Bearer {api_key}'}
            data = {
                'template': template_id,
                'layers': {
                    'shape-2': {},
                    'image-2': {
                        'image_url': f'data:image/jpeg;base64,{bg_data}',
                        'width': 846,
                        'height': 541,
                    },
                    'shape-1': {},
                    'text-3': {
                        'text': banner_title,
                        'color': '#2f4f4f',
                        'text_align': 'center',
                        'vertical_align': 'middle'
                    }
                }
            }

            resp = requests.post(url, json=data, headers=headers)
            if resp.status_code == 200:
                out = resp.json()
                if 'url' in out:
                    # download, cache, and display
                    img_bytes = requests.get(out['url']).content
                    save_banner(video_path, img_bytes)
                    st.image(img_bytes, caption=banner_title)
                    st.download_button("⬇ Download Banner", 
                                       img_bytes, 
                                       "banner.png", 
                                       "image/png",
                                        type="primary")
                else:
                    st.error("Missing image URL in response")
                    st.json(out)
            else:
                st.error(f"Render failed ({resp.status_code})")
                st.code(resp.text)


    # For downloading json file
    with tab4:  
        # Always read from persisted scene_records to avoid rerun clearing scenes
        if st.session_state.get('scene_records'):
            json_data = json.dumps(st.session_state['scene_records'], indent=2)
            st.download_button(
                "📥 Download Scene Analysis (JSON)",
                data=json_data,
                file_name="video_analysis.json",
                mime="application/json",
                key="download_scene_records"
            )
        else:
            st.info("No scene records available. Analyze scenes first.")


    with tab5:
        st.header("🎥 Promo Clip")
        if st.button("Generate Promo", key="gen_promo"):
            # First, check cache
            if st.session_state.selected_scenes:
                st.subheader("Compile Selected Scenes into Promo")
                with st.spinner("Rendering promo clip..."):
                    clips = []
                    for idx in st.session_state.selected_scenes:
                        sc = st.session_state.scenes[idx]
                        clips.append(VideoFileClip(st.session_state.uploaded).subclipped(sc['start'], sc['end']))
                    promo = concatenate_videoclips(clips)
                    out_path = promo_path(st.session_state.uploaded)
                    promo.write_videofile(out_path, codec="libx264", audio_codec="aac")
                    st.success("Promo generated")
                    st.video(out_path)
                    with open(out_path, "rb") as f:
                        btn_data = f.read()
                    st.download_button("Download Promo Clip", data=btn_data,
                                    file_name=os.path.basename(out_path), mime="video/mp4")
            else:
                st.info('Select scenes for promo generation')

    with tab6:
        if st.session_state.get("full_summary"):
            st.subheader("Full Video Summary")
            st.write(st.session_state["full_summary"])
        else:
            if not st.session_state.get('scene_records'):
                st.info("Please process scenes first.")
            else:
                st.header("Generate Full Video Summary")
                if st.button("🔄 Summarize Full Video"):
                    # try cache
                    cached = load_summary(video_path)
                    if cached:
                        full_summary = cached
                        st.success("Loaded summary from cache.")
                    else:
                        full_text = st.session_state.get("transcript", "")
                        full_text = st.session_state.get("transcript", "")
                        scene_summaries = [r["summary"] for r in st.session_state["scene_records"]]
                        chunk_summaries = summarize_chunks_with_gpt4(client, scene_summaries)
                        full_summary = final_summary_with_openai(chunk_summaries, full_text)
                        save_summary(video_path, full_summary)
                        st.success("Generated and saved summary.")
                    st.session_state["full_summary"] = full_summary   # addded this one 
                    st.subheader("Full Video Summary")
                    st.write(full_summary)
        

    # Cleanup
    st.button("🧹 Clear All", type="primary", on_click=lambda: [
        shutil.rmtree(temp_dir, ignore_errors=True),
        st.session_state.clear()
    ])
