import streamlit as st

st.set_page_config(
    page_title="Multimodal Processor",
    page_icon="📁",
    layout="wide"
)

st.title("📁 Multimodal File Processor")
st.markdown("""
    Welcome to the Multimodal File Processor! Select a modality from the sidebar to get started.
    
    - 🎨 Vision: Process image files (processing continues if you switch pages)
    - 🎵 Audio: Process audio files (processing continues if you switch pages)
    - 🎥 Video: Process video files (processing continues if you switch pages)
""")

st.sidebar.success("Select a modality above.")