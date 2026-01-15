#!/usr/bin/env python3
"""
YOLO Video Detection Script
Loads YOLO model and processes videos with object detection,
saving results with bounding boxes and class labels.
"""

import os
from pathlib import Path
from ultralytics import YOLO
import argparse
import cv2

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
WINDOW_NAME = "Inference"
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


def process_video(model_path, video_path):
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Load YOLO model
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(str(model_path))
    print("Model loaded successfully!")
    
    # Process each video
    try:
        # Run inference on video
        # show=False prevents displaying the video
        # save=True saves the output
        results = model.predict(
            source=str(video_path),
            save=False,  # We'll handle saving manually for custom output path
            show=False,  # Don't display video during processing
            conf=CONFIDENCE_THRESHOLD,   # Confidence threshold
            iou=IOU_THRESHOLD,    # IoU threshold for NMS
            stream=True  # Stream results for memory efficiency
        )
        
        # Open video capture
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        
        print(f"  Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        frame_count = 0
        for result in results:
            # Get the original frame with detections plotted
            frame = result.plot(
                conf=True,      # Show confidence scores
                labels=True,    # Show class labels
                boxes=True,     # Show bounding boxes
                line_width=2    # Box line width
            )
            
            display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(WINDOW_NAME, display_frame)
            
            # Handle keyboard input - use longer waitKey for more responsive display
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n✗ Stopped by user")
                break

            elif key == ord('p'):
                paused = not paused
                if paused:
                    print("⏸ Paused - Press 'p' to resume")
                else:
                    print("▶ Resumed")
            
            frame_count += 1
            if frame_count % 30 == 0:  # Progress update every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"  Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
        
        # Release resources
        cap.release()
        
    except Exception as e:
        print(f"  ✗ Error processing {video_path.name}: {str(e)}")
    
if __name__ == "__main__":
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    parser = argparse.ArgumentParser(
        description='Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model-path', type=str, required=True, help="Model path")
    parser.add_argument('--video-path', type=str, required=True, help="Video path")

    args = parser.parse_args()
    print("="*50)
    print("YOLO Video Detection Script")
    print("="*50)
    
    process_video(args.model_path, args.video_path)