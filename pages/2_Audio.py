import streamlit as st
import json
import io
from pathlib import Path
from audio_pipeline2 import get_audio_data

def process_audio_file(filename):
    """Wrapper function for audio processing to be run in background"""
    try:
        # Save the file temporarily if needed by your audio processing
        # Or modify get_audio_data to work with bytes directly
        return get_audio_data(filename)
    except Exception as e:
        raise Exception(f"Audio processing error: {str(e)}")

def display_audio_preview(file_obj):
    """Display audio preview with error handling"""
    if file_obj is None:
        st.warning("No file uploaded")
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

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🎵 Audio File Processor")

# Initialize session state variables
if 'processing_result' not in st.session_state:
    st.session_state.processing_result = None
if 'status' not in st.session_state:
    st.session_state.status = None

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Audio")
    file = st.file_uploader(
        "Upload audio file",
        type=['mp3', 'wav', 'ogg', 'flac', 'm4a'],
        key="audio_uploader"
    )
    
    if file is not None:
        display_audio_preview(file)
    # Process button
    if st.button("Process File", type="primary"):
        if file is not None:
            with st.spinner(f"Processing {file.name}..."):
                result = process_audio_file(file)
                st.session_state.processing_result = result
                st.session_state.status = f"✅ Successfully processed file: {file.name}"
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
    
    
    # Display results if available
    if st.session_state.processing_result:
        st.json(st.session_state.processing_result)
        
        st.download_button(
            label="Download Audio Analysis (JSON)",
            data=json.dumps(st.session_state.processing_result, indent=2),
            file_name=f"audio_results_{file.name.split('.')[0]}.json",
            mime="application/json",
            key="audio_download"
        )
    elif not st.session_state.status:
        st.info("No results to display yet. Upload an audio file and click 'Process Audio'.")
