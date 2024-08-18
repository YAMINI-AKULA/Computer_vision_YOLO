import os
import cv2
import tempfile
import streamlit as st
from tqdm import tqdm
from ultralytics import YOLO
from huggingface_hub import hf_hub_download, login

def load_model_with_token(repo_id, model_filename):
    # Get the Hugging Face access token from environment variable
    access_token = os.getenv("HF_ACCESS_TOKEN")
    
    if access_token is None:
        raise ValueError("Access token not found. Please set the HF_ACCESS_TOKEN environment variable.")

    # Authenticate using the access token
    login(token=access_token)

    # Download the model file using the token and cache it locally
    cached_model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, token=access_token)

    # Rename the file to have a .pt extension if necessary
    new_cached_model_path = f"{cached_model_path}.pt"
    os.rename(cached_model_path, new_cached_model_path)

    print(f"Downloaded model to {new_cached_model_path}")

    # Load the model using YOLO from the cached model file
    return YOLO(new_cached_model_path)

def process_video(video_path, output_path, model, conf_threshold=0.2):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in tqdm(range(total_frames), desc="Processing video"):
        success, frame = cap.read()
        if not success:
            break

        # Perform inference on the frame
        results = model(frame, conf=conf_threshold)
        
        # Annotate the frame with detection results
        annotated_frame = results[0].plot()
        
        # Write the annotated frame to the output video
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")

# Streamlit app code
st.title("Soccer Ball Detection using YOLOv8")

# Upload video file
uploaded_video = st.file_uploader("Choose a video.The model has been trained and fine-tuned to detect soccer ball in video...", type=["mp4", "avi", "mov"])

# Load YOLO model
repo_id = "Yaku03/YOLO_Soccerball"  # Replace with the actual repo ID or Space ID
model_filename = "ball_detection_model.pt"  # Replace with the actual model filename

model = load_model_with_token(repo_id, model_filename)

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    temp_video_path = tempfile.NamedTemporaryFile(delete=False).name
    with open(temp_video_path, 'wb') as f:
        f.write(uploaded_video.getbuffer())
    
    # Create a temporary file to save the processed video
    temp_output_video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

    # Process the video and get the output path
    process_video(temp_video_path, temp_output_video_path, model, conf_threshold=0.2)

    # Display the processed video
    st.video(temp_output_video_path)

    # Clean up temporary files after displaying the video
    os.remove(temp_video_path)
    os.remove(temp_output_video_path)