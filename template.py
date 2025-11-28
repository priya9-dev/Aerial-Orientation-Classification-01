import os

# Project structure dictionary
project_structure = {
    "aerial_detection": {
        "data": {
            "classification_dataset": {
                "train": {"bird": {}, "drone": {}},
                "valid": {},
                "test": {}
            },
            "object_detection_dataset": {}
        },
        "notebooks": {
            "01_data_exploration.ipynb": "",
            "02_preprocessing.ipynb": "",
            "03_model_training.ipynb": "",
            "04_yolov8_detection.ipynb": "",
        },
        "models": {},
        "src": {
            "preprocessing.py": """# preprocessing.py
# Add all preprocessing functions here

def load_images(path):
    pass

def normalize_images(images):
    pass
""",

            "model_builder.py": """# model_builder.py
# Build CNN / Transfer Learning models here

def build_cnn_model():
    pass

def build_transfer_model():
    pass
""",

            "train.py": """# train.py
# Training pipeline for classification

def train_classification_model():
    pass
""",

            "evaluate.py": """# evaluate.py
# Evaluation metrics and visualization

def evaluate_model(model, test_loader):
    pass
"""
        },
        "app.py": """# app.py (Streamlit App)
import streamlit as st

st.title("Aerial Object Detection & Classification")
st.write("Upload an image to classify Bird vs Drone or run YOLO Detection.")
""",
        "requirements.txt": """opencv-python
tensorflow
torch
ultralytics
numpy
pandas
matplotlib
seaborn
streamlit
scikit-learn
""",
        "README.md": """# Aerial Object Classification & Detection

This project includes:
- Bird vs Drone Classification (CNN / Transfer Learning)
- YOLOv8 Object Detection
- Streamlit Web App

Project Structure:
- data/ : datasets
- notebooks/ : exploratory Jupyter notebooks
- src/ : preprocessing, model building, training, evaluation scripts
- app.py : Streamlit application
"""
    }
}

# Function to create folders and files
def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)

        # If content is a dict → folder
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            # Create file
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Created file: {path}")


# Run the structure creation
create_structure(".", project_structure)

print("✅ Project structure created successfully!")

