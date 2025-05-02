import streamlit as st
import json, tempfile, os
import io
from PIL import Image
from vision_pipeline import VisionMetaData
from pathlib import Path

# Configuration & Clients Setup
# Google Cloud Setup
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
openai_apikey=st.secrets["OPENAI_API_KEY"]
aws_access_key=st.secrets["AWS_ACCESS_KEY"]
aws_secret_key=st.secrets["AWS_SECRET_KEY"]
# Initialize the vision processor

# Create results directory if it doesn't exist
RESULTS_DIR = "image_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

imageDataExtractor = VisionMetaData(credentials_path=temp_file_path, 
                                    openai_api_key=openai_apikey,
                                    aws_access_key=aws_access_key,
                                    aws_secret_key=aws_secret_key)

def display_image_preview(file_obj):
    if file_obj is None:
        st.warning("No file uploaded")
        return
    
    try:
        file_bytes = file_obj.getvalue()
        if len(file_bytes) == 0:
            st.error("Uploaded file is empty")
            return
            
        try:
            image = Image.open(io.BytesIO(file_bytes))
            st.image(image, caption=file_obj.name, use_container_width=True)
        except Exception as img_error:
            st.error(f"Couldn't display image: {str(img_error)}")
    except Exception as e:
        st.error(f"Error displaying preview: {str(e)}")


def get_result_filename(image_filename):
    """Generate the result filename with audio tags directory"""
    base_name = Path(image_filename).stem
    return os.path.join(RESULTS_DIR, f"{base_name}.json")

def load_existing_result(image_filename):
    """Check if result already exists and load it"""
    result_file = get_result_filename(image_filename)
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            return json.load(f)
    return None

def save_result(image_filename, result):
    """Save the result to a JSON file with same name as image"""
    result_file = get_result_filename(image_filename)
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)


def process_vision_file(file_obj):
    if file_obj is None:
        return {"error": "No file uploaded"}
    
    try:
        # First check if we already have results for this file
        result_file = get_result_filename(file_obj.name)
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                return json.load(f)
        
        result = imageDataExtractor.get_image_metadata(file_obj.getvalue())
        # Save the result for future use
        save_result(file_obj.name, result)
        return result
    except Exception as e:
        return {"error": str(e)}

# Streamlit UI
# Streamlit UI
st.set_page_config(layout="wide")
st.title("🎨 Vision File Processor")

# Change all st.session_state references to use 'audio_' prefix
if 'audio_processing_result' not in st.session_state:
    st.session_state.audio_processing_result = None
if 'audio_status' not in st.session_state:
    st.session_state.audio_status = None


col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    file = st.file_uploader(
        "Upload image file",
        type=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'],
        key="vision_uploader"
    )
    
    if file:
        display_image_preview(file)
    
    if st.button("Process Image", type="primary", key="vision_process"):
        if file is not None:
            with st.spinner(f"Processing {file.name}..."):
                result = process_vision_file(file)
                st.session_state.audio_processing_result = result
                st.session_state.audio_status = f"✅ Successfully processed file: {file.name}"
            st.rerun()  # Refresh to show results
        else:
            st.session_state.audio_status = "⚠️ Please upload a file first"
            st.rerun()

with col2:
    st.subheader("Processing Results")
    
    # Check processing status
    if st.session_state.audio_status:
        if st.session_state.audio_status.startswith("✅"):
            st.success(st.session_state.audio_status)
        else:
            st.warning(st.session_state.audio_status)
    
    if st.session_state.audio_processing_result:
        st.json(st.session_state.audio_processing_result)
        
        st.download_button(
            label="Download Results as JSON",
            data=json.dumps(st.session_state.audio_processing_result, indent=2),
            file_name="vision_results.json",
            mime="application/json",
            key="vision_download"
        )
    elif not st.session_state.audio_status:
        st.info("No results to display yet. Upload an image and click 'Process Image'.")
