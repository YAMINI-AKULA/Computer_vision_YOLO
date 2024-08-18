# Soccer Ball Detection Using YOLOv8 computer vision project

[Try the app form the Streamlit link](https://computervisionyolo-bdmcgrin4aez77igpctmph.streamlit.app/)

[Access the hugging face model from here](https://huggingface.co/spaces/Yaku03/YOLO_SOCCER_BALL_DETECTION)

## Demo

![image](https://github.com/user-attachments/assets/b12b4f34-b2f8-4be9-838c-14142a122507)

## Overview

This computer vision project implements a real-time soccer ball detection system using the YOLOv8 object detection model. The application is built using the ultralytics YOLO library and deployed via a Streamlit web app hosted on Hugging Face Spaces. The model has been trained and fine-tuned to detect soccer balls in videos, leveraging a custom dataset and state-of-the-art deep learning techniques. The Adam optimizer was employed to efficiently update the model weights during training, taking advantage of adaptive learning rates for each parameter.The model performance has been evaluated using key metrics such as Intersection over Union (IoU) of .83 and object detection losses.

## Technologies Used
        .	YOLOv8: The latest version of the YOLO (You Only Look Once) object detection model, known for its speed and accuracy.
	•	Streamlit: A fast and easy-to-use framework for deploying machine learning models as web applications.
	•	Hugging Face Hub: Used for storing and accessing the pre-trained model.
	•	Hugging Face Spaces: The deployment platform for the Streamlit app, providing seamless hosting and management.
	•	OpenCV: A powerful library for real-time computer vision tasks, used here for video processing.
	•	TQDM: Provides a visual progress bar for tracking the video processing status.
	•	Python: The core programming language used to implement and glue together all components.
  ## Flow of the Project

	1.	Upload Video: The user uploads a video through the Streamlit interface.
	2.	Model Inference: The video is processed frame-by-frame using the YOLOv8 model to detect soccer balls.
	3.	Video Annotation: The model’s predictions are used to annotate the video, drawing bounding boxes around detected soccer balls.
	4.	Display Output: The annotated video is rendered back to the user through the Streamlit app, showing the results of the detection

 ## Project Structure
 	•	app.py: The main application file that sets up the Streamlit app, handles video uploads, processes the video using the YOLO model, and displays the results
	•	requirements.txt: Lists all Python dependencies required to run the project.
 
 ## How It Works

	1.	Model Loading:
	•	The YOLOv8 model is loaded from the Hugging Face Model Hub using the ultralytics library.
	•	Alternatively, the model can be downloaded using hf_hub_download and loaded locally if required.
	2.	Video Processing:
	•	Users upload a video file through the Streamlit interface.
	•	The video is saved temporarily, and each frame is processed by the YOLO model to detect soccer balls.
	•	Detection results are annotated on each frame, and the processed video is saved.
	3.	Displaying Results:
	•	The processed video is displayed directly on the Streamlit interface.
	•	Users can download or view the video with the detected soccer balls highlighted.



	
