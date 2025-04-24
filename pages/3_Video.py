import streamlit as st
import tempfile, os, shutil, time
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from google.cloud import speech_v1p1beta1 as speech
import google.generativeai as genai
from transcription import transcribe
from promo import detect_scenes , save_frame , analyze_frame, summarize_scene
from banner import create_banner, extract_clip


# Settingup Configuration & Clients
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "demoproject-455507-4848ed3c5d27.json"


st.title("🎬 AI-Powered Video Analyzer")

# Defining Helper (experimenting local temp file)
def save_temp_video(uploaded):
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, uploaded.name)
    with open(path, "wb") as f: f.write(uploaded.read())
    return path, temp_dir



# Innitializing Session-State 
if 'uploaded' not in st.session_state:
    st.session_state.update({
        'uploaded': None,
        'transcript': "",
        'scene_data': [],
        'video_title': ""
    })

# UI: Upload & Buttons
uploaded = st.file_uploader("Upload a video (30s–4min)", type=["mp4","mov","avi"])
if uploaded:
    st.session_state.uploaded, tmp = save_temp_video(uploaded)

    if st.button("Transcribe Video"):
        st.session_state.transcript = transcribe(st.session_state.uploaded)

    if st.button("Detect & Analyze Scenes"):
        shots = detect_scenes(st.session_state.uploaded)
        data = []
        for i, s in enumerate(shots):
            stime, etime = s.start_time_offset.total_seconds(), s.end_time_offset.total_seconds()
            fp = os.path.join(tmp, f"frame_{i}.jpg")
            save_frame(st.session_state.uploaded, (stime+etime)/2, fp)
            with open(fp, "rb") as f:
                lbls = analyze_frame(f.read())
            words = st.session_state.transcript.split()
            sub = " ".join(words[i*50:(i+1)*50])
            sm = summarize_scene(sub, lbls)
            data.append({"start": stime, "end": etime, "frame": fp, "labels": lbls, "summary": sm})
            time.sleep(1)  
        st.session_state.scene_data = data

    # Tabs: Transcript / Analysis / Promo / Banner
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Transcript", "🎞 Scene Analysis",
        "🎥 Promo Clip", "🖼 Banner Generator"
    ])

    with tab1:
        st.header("Full Transcript")
        if st.session_state.transcript:
            st.write(st.session_state.transcript)
        else:
            st.info("Click **Transcribe Video** first.")

    with tab2:
        st.header("Scene Breakdowns")
        for i, sc in enumerate(st.session_state.scene_data):
            with st.expander(f"Scene {i+1} ({sc['start']:.1f}s–{sc['end']:.1f}s)"):
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.image(sc["frame"], width=200)
                    st.write(f"**Labels:** {', '.join(sc['labels'])}")
                with c2:
                    st.write(f"**Summary:** {sc['summary']}")

    with tab3:
        st.header("Promo Clip Creator")
        query = st.text_input("Search scenes for:")
        if query:
            matches = [
                s for s in st.session_state.scene_data
                if query.lower() in " ".join(s["labels"]).lower()
                or query.lower() in s["summary"].lower()
            ]
            sel = st.selectbox("Matches", matches,
                               format_func=lambda x: f"{x['start']:.1f}s–{x['end']:.1f}s")
            length = st.slider("Clip length", 5, 60, 30)
            if st.button("Extract Clip"):
                cs = max(sel["start"] - 5, 0)
                ce = min(cs + length, sel["end"])
                clip = os.path.join(tmp, "promo.mp4")
                extract_clip(st.session_state.uploaded, cs, ce, clip)
                st.video(clip)
                st.download_button("Download Clip", open(clip, "rb"), "promo.mp4")

    
    with tab4:
        st.header("Banner / Thumbnail Generator")
        keyword = st.text_input("Enter label to locate in video (e.g., 'land')")
        style = st.selectbox("Banner Style", ["Hero Banner","Promo Banner","Social Media"])
        if st.button("Generate Banner"):
            
            match = next(
                (
                    s for s in st.session_state.scene_data
                    if any(keyword.lower() in lbl.lower() for lbl in s['labels'])
                    or keyword.lower() in s['summary'].lower()
                ),
                None
            )
            if not match:
                st.error(f"No scene found containing '{keyword}'.")
            else:
                
                ts_mid = (match['start'] + match['end']) / 2
                img_url, status_url = create_banner(
                    st.session_state.uploaded,
                    ts_mid,
                    style
                )
                if img_url:
                    st.image(img_url)
                    st.write(f"Banner URL: {img_url}")
                else:
                    st.info(f"Banner generation in progress. Check status: {status_url}")



    # Cleanup button
    if st.button("🗑️ Clear All"):
        shutil.rmtree(tmp, ignore_errors=True)
        for k in ["uploaded", "transcript", "scene_data", "video_title"]:
            st.session_state.pop(k, None)
