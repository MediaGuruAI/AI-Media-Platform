import streamlit as st
import json, tempfile, os
import io
from PIL import Image
from vision_pipeline import VisionMetaData
from pathlib import Path

# Page Config - Set initial sidebar state to collapsed
st.set_page_config(
    page_title="Vision Processor",
    page_icon="🎨",
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
    
    [data-testid="stMarkdownContainer"] p{
    color: white;        
    }

    /* Title styling */
    .title {
        font-size: 2.5rem !important;
        font-weight: 800;
        background: linear-gradient(90deg, #ff6b6b, #ffa3a3);
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
        border: 2px solid #ff6b6b !important;
        background: transparent !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        background: #ff6b6b !important;
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

# Configuration & Clients Setup
google_credentials = st.secrets["google_cloud"]
credentials_dict = {
    "type": "service_account",
    "project_id": google_credentials["project_id"],
    "private_key_id": google_credentials["private_key_id"],
    "private_key": google_credentials["private_key"].replace('\\n', '\n'),
    "client_email": google_credentials["client_email"],
    "client_id": google_credentials["client_id"],
    "auth_uri": google_credentials["auth_uri"],
    "token_uri": google_credentials["token_uri"],
    "auth_provider_x509_cert_url": google_credentials["auth_provider_x509_cert_url"],
    "client_x509_cert_url": google_credentials["client_x509_cert_url"]
}

with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as temp_file:
    json.dump(credentials_dict, temp_file)
    temp_file_path = temp_file.name

openai_apikey = st.secrets["OPENAI_API_KEY"]
aws_access_key = st.secrets["AWS_ACCESS_KEY"]
aws_secret_key = st.secrets["AWS_SECRET_KEY"]

# Initialize the vision processor
RESULTS_DIR = "image_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

imageDataExtractor = VisionMetaData(
    credentials_path=temp_file_path, 
    openai_api_key=openai_apikey,
    aws_access_key=aws_access_key,
    aws_secret_key=aws_secret_key
)

def display_image_preview(file_obj):
    if file_obj is None:
        return
    
    try:
        file_bytes = file_obj.getvalue()
        if len(file_bytes) == 0:
            st.error("Uploaded file is empty")
            return
            
        try:
            image = Image.open(io.BytesIO(file_bytes))
            st.image(image, caption=file_obj.name, use_container_width=True)  # Changed from use_column_width
        except Exception as img_error:
            st.error(f"Couldn't display image: {str(img_error)}")
    except Exception as e:
        st.error(f"Error displaying preview: {str(e)}")

def get_result_filename(image_filename):
    base_name = Path(image_filename).stem
    return os.path.join(RESULTS_DIR, f"{base_name}.json")

def load_existing_result(image_filename):
    result_file = get_result_filename(image_filename)
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            return json.load(f)
    return None

def save_result(image_filename, result):
    result_file = get_result_filename(image_filename)
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

def process_vision_file(file_obj):
    if file_obj is None:
        return {"error": "No file uploaded"}
    
    try:
        result_file = get_result_filename(file_obj.name)
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                return json.load(f)
        
        result = imageDataExtractor.get_image_metadata(file_obj.getvalue())
        save_result(file_obj.name, result)
        return result
    except Exception as e:
        return {"error": str(e)}

# Main UI
st.markdown('<h1 class="title">🎨 Vision File Processor</h1>', unsafe_allow_html=True)

# Back button
if st.button("← Back to Main Page"):
    st.session_state.processing_result = None
    st.session_state.processing_status = "⚠️ Please upload a file first"
    st.switch_page("app.py")

# Initialize session state
if 'processing_result' not in st.session_state:
    st.session_state.processing_result = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = None

# Two-column layout
col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.subheader("Upload Image")
        file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'],
            key="vision_uploader",
            label_visibility="collapsed"
        )
        
        if file:
            display_image_preview(file)
        
        if st.button("Process Image", type="primary", key="vision_process"):
            if file is not None:
                with st.spinner(f"Processing {file.name}..."):
                    result = process_vision_file(file)
                    st.session_state.processing_result = result
                    st.session_state.processing_status = f"✅ Successfully processed file: {file.name}"
                st.rerun()
            else:
                st.session_state.processing_status = "⚠️ Please upload a file first"
                st.rerun()

with col2:
    with st.container():
        st.subheader("Processing Results")
        
        if st.session_state.processing_status:
            if st.session_state.processing_status.startswith("✅"):
                st.success(st.session_state.processing_status)
            else:
                st.warning(st.session_state.processing_status)
        
        if st.session_state.processing_result:
            st.json(st.session_state.processing_result)
            
            st.download_button(
                label="Download Results as JSON",
                data=json.dumps(st.session_state.processing_result, indent=2),
                file_name="vision_results.json",
                mime="application/json",
                key="vision_download",
                type="primary"
            )
        elif not st.session_state.processing_status:
            st.info("No results to display yet. Upload an image and click 'Process Image'.")