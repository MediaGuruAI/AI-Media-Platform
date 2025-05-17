import streamlit as st
from streamlit_extras.stylable_container import stylable_container

# Page Config with initially collapsed sidebar
st.set_page_config(
    page_title="Multimodal Processor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapsed by default but can be toggled
)

# Custom CSS to style the sidebar and toggle button
st.markdown("""
<style>
    /* Main page styling */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    
    /* Sidebar styling - initially hidden but can be revealed */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.9) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Sidebar navigation items */
    [data-testid="stSidebarNav"] li {
        padding: 10px 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    [data-testid="stSidebarNav"] li:hover {
        background: rgba(74, 58, 255, 0.2);
    }
    
    /* Title styling */
    .modern-title {
        font-size: 3rem !important;
        font-weight: 800;
        background: linear-gradient(90deg, #4a3aff, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Card styling */
    .card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        padding: 25px;
        height: 100%;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(74, 58, 255, 0.3);
    }
    
    /* Button styling */
    .modality-btn {
        width: 100%;
        margin-top: 15px;
        border: 2px solid #4a3aff !important;
        background: transparent !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    .modality-btn:hover {
        background: #4a3aff !important;
    }
    
    /* Toggle button styling */
    .sidebar-toggle {
        position: fixed;
        left: 10px;
        top: 10px;
        z-index: 999999;
        background: rgba(15, 12, 41, 0.9) !important;
        border: 1px solid #4a3aff !important;
        color: white !important;
    }
    
    .sidebar-toggle:hover {
        background: #4a3aff !important;
    }
</style>
""", unsafe_allow_html=True)

# # Add a sidebar toggle button
# toggle = st.button("☰", key="sidebar_toggle", help="Toggle Sidebar")

# # Toggle sidebar state
# if toggle:
#     current_state = st.session_state.get("sidebar_state", "collapsed")
#     if current_state == "collapsed":
#         st.session_state.sidebar_state = "expanded"
#     else:
#         st.session_state.sidebar_state = "collapsed"
    
# Set sidebar state from session
if "sidebar_state" in st.session_state:
    st.set_page_config(initial_sidebar_state=st.session_state.sidebar_state)

# Main content
st.markdown('<h1 class="modern-title">🔮 Multimodal AI Processor</h1>', unsafe_allow_html=True)

st.markdown("""
    <div style='color: rgba(255, 255, 255, 0.8); font-size: 1.1rem; margin-bottom: 2rem;'>
    Advanced processing for multiple data modalities powered by AI. 
    Select a modality below to begin your analysis.
    </div>
""", unsafe_allow_html=True)

# Modality selection cards
col1, col2, col3 = st.columns(3)

with col1:
    with stylable_container(
        key="vision_card",
        css_styles="""
            {
                border: 1px solid rgba(255, 107, 107, 0.3);
                border-radius: 15px;
            }
            h3 {
                color: #ff6b6b;
            }
        """
    ):
        with st.container():
            st.markdown("### 🎨 Vision Processing")
            st.markdown("""
            - Object detection
            - Image classification
            - Style transfer
            """)
            if st.button("Select Vision", key="vision_btn"):
                st.switch_page("pages/1_Vision.py")

with col2:
    with stylable_container(
        key="audio_card",
        css_styles="""
            {
                border: 1px solid rgba(78, 205, 196, 0.3);
                border-radius: 15px;
            }
            h3 {
                color: #4ecdc4;
            }
        """
    ):
        with st.container():
            st.markdown("### 🎵 Audio Processing")
            st.markdown("""
            - Speech recognition
            - Sound classification
            - Audio enhancement
            """)
            if st.button("Select Audio", key="audio_btn"):
                st.switch_page("pages/2_Audio.py")

with col3:
    with stylable_container(
        key="video_card",
        css_styles="""
            {
                border: 1px solid rgba(255, 159, 28, 0.3);
                border-radius: 15px;
            }
            h3 {
                color: #ff9f1c;
            }
        """
    ):
        with st.container():
            st.markdown("### 🎥 Video Processing")
            st.markdown("""
            - Action recognition
            - Object tracking
            - Video summarization
            """)
            if st.button("Select Video", key="video_btn"):
                st.switch_page("pages/3_Video.py")