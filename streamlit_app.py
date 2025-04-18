import streamlit as st
import json
import time
from pathlib import Path
from PIL import Image
import io
import warnings
from vision_pipeline import VisionMetaData
from audio_pipeline2 import get_audio_data
from pydub import AudioSegment


imageDataExtractor = VisionMetaData(credentials_path="E:\\my_documents\\demoproject-455507-4848ed3c5d27.json")


def process_file(modality: str, file_obj):
    """Simulate file processing"""
    
    if file_obj is None:
        return {"error": "No file uploaded"}
    
    file_name = file_obj.name
    file_size = len(file_obj.getvalue())
    
    if modality == "vision":
        imageData = imageDataExtractor.get_image_metadata(file_obj.getvalue())
        return imageData
    
    elif modality == "audio":
        audio_content = file_obj.getvalue()
        filename = file_obj.name
        # audio = AudioSegment.from_file(io.BytesIO(audio_content))
        audioData = get_audio_data(filename)
        return audioData
    
    elif modality == "video":
        return {
            "modality": "video",
            "file_info": {
                "name": file_name,
                "size": f"{file_size} bytes",
                "type": "video"
            },
            "analysis": {
                "format": file_name.split('.')[-1],
                "duration": "2:30",
                "fps": 30
            }
        }
    else:
        return {"error": "Unknown modality"}


def display_file_preview(file_obj, modality):
    """Display file preview with robust error handling"""
    if file_obj is None:
        st.warning("No file uploaded")
        return
    
    try:
        # First verify we can read the file
        file_bytes = file_obj.getvalue()
        if len(file_bytes) == 0:
            st.error("Uploaded file is empty")
            return
            
        file_ext = Path(file_obj.name).suffix.lower()
        
        st.subheader("File Preview")
        
        # Image preview
        if modality == "vision" or file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.jfif']:
            try:
                image = Image.open(io.BytesIO(file_bytes))
                st.image(image, caption=file_obj.name, use_container_width=True)
            except Exception as img_error:
                st.error(f"Couldn't display image: {str(img_error)}")
        
        # Audio preview
        elif modality == "audio" or file_ext in ['.mp3', '.wav', '.ogg']:
            st.audio(file_bytes, format=f'audio/{file_ext[1:]}')
        
        # Video preview
        elif modality == "video" or file_ext in ['.mp4', '.mov', '.avi']:
            st.video(file_bytes, format=f'video/{file_ext[1:]}')
        
        # Fallback for other files
        else:
            st.warning(f"No preview available for {file_ext} files")
            st.download_button(
                "Download file",
                file_bytes,
                file_obj.name
            )
            
    except AttributeError:
        st.error("Invalid file object - please upload again")
    except Exception as e:
        st.error(f"Error displaying preview: {str(e)}")


# Streamlit UI
st.set_page_config(layout="wide")  # Use wider layout
st.title("📁 Multimodal File Processor")
st.markdown("Select a modality, upload a file, and view the processed results in JSON format.")

# Initialize session state variables
if 'processing_result' not in st.session_state:
    st.session_state.processing_result = None
if 'status' not in st.session_state:
    st.session_state.status = None

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Parameters")
    
    # Modality selection
    modality = st.radio(
        "Select Modality",
        ["vision", "audio", "video"],
        index=0,
        horizontal=True
    )
    
    # File uploader with type restrictions
    file_types = {
        "vision": None,
        "audio": None,
        "video": None
    }
    
    file = st.file_uploader(
        f"Upload {modality} file",
        type=file_types[modality],
        key=f"uploader_{modality}"  # Unique key per modality
    )
    
    if file is not None:
        display_file_preview(file, modality)
    # Process button
    if st.button("Process File", type="primary"):
        if file is not None:
            with st.spinner(f"Processing {file.name}..."):
                result = process_file(modality, file)
                st.session_state.processing_result = result
                st.session_state.status = f"✅ Successfully processed {modality} file: {file.name}"
            st.rerun()  # Refresh to show results
        else:
            st.session_state.status = "⚠️ Please upload a file first"
            st.rerun()

with col2:
    st.subheader("Processing Results")
    
    # Display status message
    if st.session_state.status:
        if st.session_state.status.startswith("✅"):
            st.success(st.session_state.status)
        else:
            st.warning(st.session_state.status)
    
    # Display processing results
    if st.session_state.processing_result:
        st.json(st.session_state.processing_result)
        
        # Add download button for results
        st.download_button(
            label="Download Results as JSON",
            data=json.dumps(st.session_state.processing_result, indent=2),
            file_name="processing_results.json",
            mime="application/json"
        )
    else:
        st.info("No results to display yet. Upload a file and click 'Process File'.")