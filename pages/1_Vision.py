import streamlit as st
import json
import io
from PIL import Image
from vision_pipeline import VisionMetaData

# Initialize the vision processor
imageDataExtractor = VisionMetaData(credentials_path="streamlit_app\\demoproject-455507-4848ed3c5d27.json")

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

def process_vision_file(file_obj):
    if file_obj is None:
        return {"error": "No file uploaded"}
    
    try:
        return imageDataExtractor.get_image_metadata(file_obj.getvalue())
    except Exception as e:
        return {"error": str(e)}

# Streamlit UI
# Streamlit UI
st.set_page_config(layout="wide")
st.title("🎨 Vision File Processor")

# Initialize session state variables
if 'processing_result' not in st.session_state:
    st.session_state.processing_result = None
if 'status' not in st.session_state:
    st.session_state.status = None


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
                st.session_state.processing_result = result
                st.session_state.status = f"✅ Successfully processed file: {file.name}"
            st.rerun()  # Refresh to show results
        else:
            st.session_state.status = "⚠️ Please upload a file first"
            st.rerun()

with col2:
    st.subheader("Processing Results")
    
    # Check processing status
    if st.session_state.status:
        if st.session_state.status.startswith("✅"):
            st.success(st.session_state.status)
        else:
            st.warning(st.session_state.status)
    
    if st.session_state.processing_result:
        st.json(st.session_state.processing_result)
        
        st.download_button(
            label="Download Results as JSON",
            data=json.dumps(st.session_state.processing_result, indent=2),
            file_name="vision_results.json",
            mime="application/json",
            key="vision_download"
        )
    elif not st.session_state.status:
        st.info("No results to display yet. Upload an image and click 'Process Image'.")
