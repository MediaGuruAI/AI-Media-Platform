import streamlit as st
import json, os
from pathlib import Path
from audio_pipeline import get_audio_data

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
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
        color: white !important;
    }
    
    /* Hide the default sidebar navigation */
    [data-testid="stSidebarNav"] {
        display: none;
    }
    
    /* Title styling */
    .title {
        font-size: 2.5rem !important;
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

# Configuration
openai_api_key = st.secrets["OPENAI_API_KEY"]
azure_key = st.secrets['AZURE_API_KEY']
azure_region = st.secrets['AZURE_API_REGION']

# Create results directory if it doesn't exist
RESULTS_DIR = "audio_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_result_filename(audio_filename):
    """Generate the result filename with audio tags directory"""
    base_name = Path(audio_filename).stem
    return os.path.join(RESULTS_DIR, f"{base_name}.json")

def process_audio_file(file_obj):
    """Wrapper function for audio processing to be run in background"""
    try:
        # Check if result already exists
        result_filename = get_result_filename(file_obj.name)
        if os.path.exists(result_filename):
            print('Loading from local storage')
            with open(result_filename, 'r') as f:
                return json.load(f)
        
        # Process the audio file
        result = get_audio_data(file_obj, openai_api_key, azure_key, azure_region)
        
        # Save the result
        with open(result_filename, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
    except Exception as e:
        raise Exception(f"Audio processing error: {str(e)}")

def display_audio_preview(file_obj):
    """Display audio preview with error handling"""
    if file_obj is None:
        return
    
    try:
        file_bytes = file_obj.getvalue()
        if len(file_bytes) == 0:
            st.error("Uploaded file is empty")
            return
            
        file_ext = Path(file_obj.name).suffix.lower()
        st.audio(file_bytes, format=f'audio/{file_ext[1:]}')
    except Exception as e:
        st.error(f"Error displaying preview: {str(e)}")

# Main UI
st.markdown('<h1 class="title">🎵 Audio File Processor</h1>', unsafe_allow_html=True)

# Back button
if st.button("← Back to Main Page"):
    st.switch_page("app.py")

# Initialize session state variables
if 'processing_result' not in st.session_state:
    st.session_state.processing_result = None
if 'status' not in st.session_state:
    st.session_state.status = None

col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.subheader("Upload Audio")
        file = st.file_uploader(
            "Choose an audio file",
            type=['mp3', 'wav', 'ogg', 'flac', 'm4a'],
            key="audio_uploader",
            label_visibility="collapsed"
        )
        
        if file is not None:
            display_audio_preview(file)
        
        if st.button("Process File", type="primary"):
            if file is not None:
                with st.spinner(f"Processing {file.name}..."):
                    try:
                        result = process_audio_file(file)
                        st.session_state.processing_result = result
                        st.session_state.status = f"✅ Successfully processed file: {file.name}"
                    except Exception as e:
                        st.session_state.status = f"❌ Error processing file: {str(e)}"
                st.rerun()
            else:
                st.session_state.status = "⚠️ Please upload a file first"
                st.rerun()

with col2:
    with st.container():
        st.subheader("Processing Results")
        
        if st.session_state.status:
            if st.session_state.status.startswith("✅"):
                st.success(st.session_state.status)
            elif st.session_state.status.startswith("❌"):
                st.error(st.session_state.status)
            else:
                st.warning(st.session_state.status)
        
        if st.session_state.processing_result:
            st.json(st.session_state.processing_result)
            
            st.download_button(
                label="Download Results as JSON",
                data=json.dumps(st.session_state.processing_result, indent=2),
                file_name=f"audio_results_{Path(file.name).stem}.json",
                mime="application/json",
                key="audio_download"
            )
        elif not st.session_state.status:
            st.info("No results to display yet. Upload an audio file and click 'Process Audio'.")