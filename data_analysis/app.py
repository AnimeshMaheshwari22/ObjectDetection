from pathlib import Path
import streamlit as st
import pandas as pd
import argparse
import sys
from pathlib import Path
from src.parser import load_labels_from_directory
from src.visualizer import (
    plot_class_distribution,
    plot_box_area_by_class,
    plot_distribution_by_condition,
    get_annotated_image
)

st.set_page_config(
    page_title="BDD100k Analysis Dashboard",
    page_icon="",
    layout="wide"
)

parser = argparse.ArgumentParser(description="BDD100k Analysis Dashboard")
parser.add_argument("--data_dir_labels", type=str, required=True, help="Path to labels directory")
parser.add_argument("--data_dir_img", type=str, required=True, help="Path to images directory")
parser.add_argument("--split", type=str, choices=["train", "val"], required=True, help="Dataset split: train or val")
args = parser.parse_args()

DATA_DIR_LABEL = Path(args.data_dir_labels)
DATA_DIR_IMG = Path(args.data_dir_img)
SPLIT_TYPE = args.split

@st.cache_data
def load_data(labels_path):
    if not labels_path.exists():
        st.error(f"Error: Label directory not found at {labels_path}. Please check the DATA_DIR path in app.py.")
        return None
    df = load_labels_from_directory(labels_path)
    return df

st.title("BDD100k Object Detection Analysis 📊")
st.write("""
This dashboard provides an interactive analysis of the BDD100k dataset.
Navigate through the sections using the sidebar to explore the data.
""")

st.sidebar.header("Dataset Configuration")
st.sidebar.write(f"Using split: **{SPLIT_TYPE}**")

LABELS_PATH = DATA_DIR_LABEL / SPLIT_TYPE
IMAGES_PATH = DATA_DIR_IMG / SPLIT_TYPE / "img"

df = load_data(LABELS_PATH)

if df is not None:
    st.sidebar.header("Navigation")
    analysis_mode = st.sidebar.radio(
        "Choose an analysis section:",
        ("Overall Distribution", "Detailed Analysis", "Interesting Samples")
    )

    if analysis_mode == "Overall Distribution":
        st.header("Overall Object Class Distribution")
        st.write("This chart shows the total number of instances for each object category, highlighting the significant class imbalance in the dataset.")
        
        fig = plot_class_distribution(df)
        st.pyplot(fig)

        
    elif analysis_mode == "Detailed Analysis":
        st.header("Detailed Analysis by Category and Condition")
        
        st.subheader("Bounding Box Area Analysis")
        st.write("Select an object class to see the distribution of its bounding box sizes (on a logarithmic scale).")
        
        selected_class = st.selectbox(
            "Select a class:",
            df['category'].unique()
        )
        if selected_class:
            fig_box = plot_box_area_by_class(df, selected_class)
            st.pyplot(fig_box)
            
        st.subheader("Conditional Distribution Analysis")
        st.write("Explore how the presence of objects changes based on environmental conditions.")
        
        selected_condition = st.selectbox(
            "Select a condition:",
            ('weather', 'scene', 'timeofday')
        )
        if selected_condition:
            fig_cond = plot_distribution_by_condition(df, selected_condition)
            st.pyplot(fig_cond)

    elif analysis_mode == "Interesting Samples":
        st.header("Visualizing Interesting Samples")
        st.write("View annotated images for specific scenarios. (Requires images to be downloaded)")

        if not IMAGES_PATH.exists():
            st.warning("Image directory not found. Please download the images and update the path in app.py to use this feature.")
        else:
            sample_type = st.selectbox(
                "Select a sample type:",
                ["Image with the most objects (High Complexity)", "Image with a rare class (train)"]
            )
            image_name_to_show = None
            if "most objects" in sample_type:
                most_complex_image = df['image_name'].value_counts().nlargest(1).index[0]
                image_name_to_show = most_complex_image
            elif "rare class" in sample_type:
                train_images = df[df['category'] == 'train']['image_name'].unique()
                if len(train_images) > 0:
                    image_name_to_show = train_images[0]
                else:
                    st.warning("No images with the 'train' class found in the dataset.")
            
            if image_name_to_show:
                st.write(f"Displaying image: **{image_name_to_show}**")
                annotated_image = get_annotated_image(image_name_to_show, IMAGES_PATH, df)
                if annotated_image is not None:
                    st.image(annotated_image, caption=f"Annotated: {image_name_to_show}", use_column_width=True)
                else:
                    st.error(f"Could not load image file for {image_name_to_show}. Make sure it exists in the images directory.")

else:
    st.warning("Data could not be loaded. Please ensure the path is set correctly in the app.py file.")