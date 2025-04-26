import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Lung Cancer Classification",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    /* Beautified Tabs */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 24px;
        background: #f8fafc;
        border-radius: 18px;
        box-shadow: 0 2px 12px rgba(30,58,138,0.07);
        padding: 0.5rem 1rem;
        margin-bottom: 2rem;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1e293b;
        background: #e0e7ef;
        border-radius: 12px;
        padding: 12px 28px;
        margin-right: 2px;
        transition: all 0.2s;
        border: none;
        box-shadow: none;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #c7d2fe;
        color: #1d4ed8;
        cursor: pointer;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, #2563eb 0%, #38bdf8 100%);
        color: #fff !important;
        box-shadow: 0 2px 8px rgba(30,58,138,0.10);
        font-weight: 700;
        border: none;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.15rem;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    .prediction-text {
        position: absolute;
        top: 10px;
        left: 10px;
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Color mapping for different classes
CLASS_COLORS = {
    "Benign case": "#2ecc71",  # Green
    "Malignant case": "#e74c3c",  # Red
    "Normal case": "#00ffdd",  # Cyan
    "Benign case Malignant case": ["#2ecc71", "#e74c3c"],  # Green + Red
    "Malignant case Normal case": ["#e74c3c", "#00ffdd"]  # Red + Cyan
}

def load_model():
    """Load the YOLOv8 model"""
    model = YOLO('best.pt')
    return model

def process_image(image, target_size=(640, 640)):
    """
    Process and resize image while maintaining aspect ratio
    """
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    elif isinstance(image, str):
        img = Image.open(image)
    else:
        img = image

    # Calculate aspect ratio
    aspect = img.width / img.height
    
    if aspect > 1:
        new_width = target_size[0]
        new_height = int(target_size[0] / aspect)
    else:
        new_height = target_size[1]
        new_width = int(target_size[1] * aspect)
        
    img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Create new image with padding
    new_img = Image.new('RGB', target_size, (240, 240, 240))
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    new_img.paste(img, (paste_x, paste_y))
    
    return np.array(new_img)

def draw_prediction(image, class_name, confidence):
    """Draw prediction on image with colored border and text"""
    img = image.copy()
    height, width = img.shape[:2]
    
    # Create border
    border_size = 10
    colors = CLASS_COLORS[class_name]
    
    if isinstance(colors, list):  # Combined classes
        # Split border into two colors
        border_img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(border_img, (0, 0), (width, height), colors[0], border_size)
        cv2.rectangle(border_img, (border_size, border_size), 
                     (width-border_size, height-border_size), colors[1], border_size)
        
        # Combine with original image
        mask = np.any(border_img != [0, 0, 0], axis=2)
        img[mask] = border_img[mask]
        
        # Add text with gradient background
        text_bg_height = 40
        gradient = np.linspace(0, 1, width).reshape(1, -1, 1)
        color1 = np.array(tuple(int(colors[0].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
        color2 = np.array(tuple(int(colors[1].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
        gradient_colors = color1 * (1 - gradient) + color2 * gradient
        text_bg = gradient_colors.astype(np.uint8)
        text_bg = cv2.resize(text_bg, (width, text_bg_height))
        
        # Add semi-transparent overlay
        overlay = img.copy()
        overlay[:text_bg_height] = text_bg
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Add text
        font = cv2.FONT_HERSHEY_DUPLEX
        text = f"{class_name} ({confidence:.1%})"
        cv2.putText(img, text, (10, 30), font, 0.8, (255, 255, 255), 2)
        
    else:  # Single class
        # Create colored border
        cv2.rectangle(img, (0, 0), (width, height), 
                     tuple(int(colors.lstrip('#')[i:i+2], 16) for i in (4, 2, 0)), 
                     border_size)
        
        # Add colored text banner
        text_bg_height = 40
        color = tuple(int(colors.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (width, text_bg_height), color, -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Add text
        font = cv2.FONT_HERSHEY_DUPLEX
        text = f"{class_name} ({confidence:.1%})"
        cv2.putText(img, text, (10, 30), font, 0.8, (255, 255, 255), 2)
    
    return img

def main():
    # Create tabs
    tabs = st.tabs(["🔍 Lung Cancer Detection", "ℹ️ Project Info", "👥 Team Information"])
    
    # Tab 1: Main Application
    with tabs[0]:
        st.title("Lung Cancer Classification")
        st.write("Upload a lung scan image for analysis")
        
        try:
            model = load_model()
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Center container for image
            with st.container():
                col1, col2, col3 = st.columns([1,3,1])
                
                with col2:
                    image = Image.open(uploaded_file)
                    # Process and resize image
                    processed_img = process_image(image)
                    
                    try:
                        # Make prediction
                        results = model.predict(processed_img)
                        pred_class = results[0].probs.top1
                        confidence = float(results[0].probs.top1conf)
                        class_name = model.names[pred_class]
                        
                        # Draw prediction
                        result_img = draw_prediction(processed_img, class_name, confidence)
                        
                        # Display result
                        st.image(result_img, use_column_width=True)
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        st.image(processed_img, use_column_width=True)

    # Tab 2: Project Info
    with tabs[1]:
        st.title("Project Information")
        
        st.header("Dataset")
        st.write("The dataset was sourced from the Roboflow Universe/Kaggle, specifically curated for lung cancer classification.")
        
        st.header("Model Training Process")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info("**Model Architecture**\nYOLOv8 Classification")
        with col2:
            st.info("**Number of Epochs**\n50")
        with col3:
            st.info("**Image Size**\n224 × 224")
        with col4:
            st.info("**Accuracy**\n95%")
            
        st.header("Classes")
        for class_name, color in CLASS_COLORS.items():
            st.markdown(f'<div style="background-color: {color}; padding: 10px; border-radius: 5px; color: white; margin: 5px 0;">{class_name}</div>', unsafe_allow_html=True)

    # Tab 3: Team Information
    with tabs[2]:
        st.title("Team Information")
        
        st.header("University")
        st.write("UET Mardan")
        
        st.header("Department")
        st.write("Telecommunication")
        
        st.header("Team Members")
        members = {
            "Asif Khan": "21MDTLE195",
            "Muhammad Ifzal": "21MDTLE206"
        }
        
        for member, reg in members.items():
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;'>
                <h3>{member}</h3>
                <p>{reg}</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.header("Supervisor")
        st.write("Dr. jalal Khan")

        st.header("Co-Supervisor")
        st.write("Prof. Dr. Khalid Khan")

if __name__ == "__main__":
    main()