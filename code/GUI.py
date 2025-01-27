import streamlit as st
import pandas as pd
import os
import cv2
import subprocess
import subprocess
import mimetypes

def try_display_video(gaze_video_path):
    """
    Tries to display a video using Streamlit. If the format is unsupported, 
    re-encodes the video to a compatible format and retries.

    Parameters:
        gaze_video_path (str): Path to the video file.
    """
    # Check the MIME type of the video
    mime_type, _ = mimetypes.guess_type(gaze_video_path)
    supported_types = ["video/mp4", "video/webm", "video/ogg"]

    if mime_type not in supported_types:
        st.warning("Unsupported video format detected. Re-encoding to MP4 for compatibility...")
        
        # Re-encode the video using FFmpeg
        reencoded_path = f"{gaze_video_path.rsplit('.', 1)[0]}_reencoded.mp4"
        command = [
            "ffmpeg",
            "-i", gaze_video_path,
            "-vcodec", "libx264",
            "-acodec", "aac",
            "-strict", "experimental",
            "-y",  # Overwrite existing file
            reencoded_path
        ]
        try:
            subprocess.run(command, check=True, capture_output=True)
            st.success("Video re-encoded successfully. Displaying video...")
            gaze_video_path = reencoded_path  # Update the path to the re-encoded video
        except subprocess.CalledProcessError as ffmpeg_error:
            st.error(f"FFmpeg re-encoding failed: {ffmpeg_error.stderr.decode() if ffmpeg_error.stderr else 'No error output.'}")
            return
        except FileNotFoundError:
            st.error("FFmpeg is not installed or not in the system's PATH.")
            return

    # Try displaying the video
    try:
        video_file = open(gaze_video_path, 'rb')
        video_bytes = video_file.read()
        st.video(gaze_video_path)
    except Exception as e:
        st.error(f"Unexpected error displaying the video: {e}")



def cut_video(input_video, output_video, start_time, duration):
    """Cuts a video using FFmpeg with millisecond precision."""
    command = [
        "ffmpeg",
        "-ss", str(start_time),
        "-i", input_video,
        "-t", str(duration),
        "-c:v", "copy",
        "-c:a", "copy",
        "-y",
        output_video,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
        print(f"Video cut successfully. Output saved to {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error during video cutting: {e}")
        print(f"FFmpeg output: {e.stderr.decode() if e.stderr else 'No error output.'}")
    except FileNotFoundError:
        print("Error: FFmpeg is not installed or not in the system's PATH.")

def get_video_duration(gaze_video_path):
    """Retrieve the duration of the video in seconds."""
    cap = cv2.VideoCapture(gaze_video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else None
    cap.release()
    return duration

def annotate_video(output_dir, annotation_csv_path):
    """Streamlit interface for annotating a video by selecting chunks and adding annotations."""
    st.title("Video Chunking and Annotation Tool")
    
    # File input for video path
    gaze_video_path = st.text_input("Input Video Path (with gaze):", "")
    no_gaze_video_path = st.text_input("Input Video Path (without gaze):", "")
    if st.button("Load Videos"):
        if os.path.exists(gaze_video_path) and os.path.exists(no_gaze_video_path):
            st.session_state.gaze_video_path = gaze_video_path
            st.session_state.no_gaze_video_path = no_gaze_video_path
            gaze_video_duration = get_video_duration(gaze_video_path)
            no_gaze_video_duration = get_video_duration(no_gaze_video_path)
            if gaze_video_duration and no_gaze_video_duration:
                st.session_state.gaze_video_duration = gaze_video_duration
                st.session_state.no_gaze_video_duration = no_gaze_video_duration
                st.success(f"Video loaded: {gaze_video_path}, {no_gaze_video_path} (Duration: {gaze_video_duration:.2f}, {no_gaze_video_duration:.2f} seconds)")
                if gaze_video_duration ==no_gaze_video_duration:
                    st.success(f"Videos are of the same duration. ")
                else:
                    st.error("Video with and without gaze are of different duration. ")
            else:
                st.error("Unable to retrieve video duration. Please check the file.")
        else:
            st.error("Invalid video path. Please enter a valid path.")

    # Check if video is loaded
    if "gaze_video_path" in st.session_state and st.session_state.gaze_video_path:
        gaze_video_path = st.session_state.gaze_video_path
        
        # Video playback
        try_display_video(gaze_video_path)
        

        # Slider for selecting start and end times
        st.markdown("### Add Annotation")
        max_duration = st.session_state.gaze_video_duration if "gaze_video_duration" in st.session_state else 300.0
        start_time, end_time = st.slider(
            "Select Start and End Times:",
            min_value=0.0,
            max_value=max_duration,
            value=(0.0, 1.0),
            step=0.001,
            format="%.3f"
        )
        
        if st.button("Preview"):
            st.markdown(f"Previewing video from {start_time:.3f} to {end_time:.3f} seconds.")
            command = [
                "ffmpeg",
                "-ss", str(start_time),
                "-i", gaze_video_path,
                "-t", str(end_time - start_time),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-y",
                "preview.mp4",
            ]
            try:
                subprocess.run(command, check=True, capture_output=True)
                preview_file = open("preview.mp4", "rb")
                preview_bytes = preview_file.read()
                st.video(preview_bytes)
            except subprocess.CalledProcessError as e:
                st.error(f"Error generating preview: {e.stderr.decode() if e.stderr else 'No error output.'}")


        # Input for annotation text
        action = st.text_input("Action (eg. Place)", "")
        target_obj = st.text_input("Object (eg. Phone)", "")
        target_furniture = st.text_input("Furniture (eg. Table)", "")
        obj = st.text_input("Other objects (eg. Rubics cube)", "")

        if st.button("Add Annotation"):
            if end_time > start_time:
                if "annotations" not in st.session_state:
                    st.session_state.annotations = []
                st.session_state.annotations.append({
                    "vid_name": os.path.splitext(os.path.basename(gaze_video_path))[0],
                    "input_vid_pth": gaze_video_path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "action": action,
                    "target_obj": target_obj,
                    "target_furniture": target_furniture,
                    "objects": obj,
                    "annotation": action + " " + target_obj
                })
                st.session_state.annotations.append({
                    "vid_name": os.path.splitext(os.path.basename(no_gaze_video_path))[0],
                    "input_vid_pth": no_gaze_video_path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "action": action,
                    "target_obj": target_obj,
                    "target_furniture": target_furniture,
                    "objects": obj,
                    "annotation": action + " " + target_obj
                })
                st.success("Annotation added!")
            else:
                st.error("End time must be greater than start time.")

        # Display current annotations
        if "annotations" in st.session_state and st.session_state.annotations:
            st.markdown("### Current Annotations")
            annotations_df = pd.DataFrame(st.session_state.annotations)
            st.write(annotations_df)

        # Export video chunks
        if st.button("Export Video Chunks"):
            os.makedirs(output_dir, exist_ok=True)
            updated_annotations = []
            # Progress bar
            progress_bar = st.progress(0)
            total_annotations = len(st.session_state.annotations)
            
            
            for i, annotation in enumerate(st.session_state.annotations):
                chunk_path = os.path.join(output_dir, f"{annotation['action']}_{annotation['vid_name']}_{i + 1}.mp4")
                duration = annotation["end_time"] - annotation["start_time"]
                cut_video(annotation["input_vid_path"], chunk_path, annotation["start_time"], duration)

                # Append chunk path to annotation
                updated_annotation = annotation.copy()
                updated_annotation["chunk_path"] = chunk_path
                updated_annotations.append(updated_annotation)
                progress_bar.progress((i + 1) / total_annotations)
                st.success(f"Exported: {chunk_path}")
            
            progress_bar.empty()  # Remove the progress bar once done
            st.success("All video chunks exported successfully!")
            # Append new annotations to the CSV file
            if os.path.exists(annotation_csv_path):
                existing_annotations = pd.read_csv(annotation_csv_path)
                updated_annotations_df = pd.DataFrame(updated_annotations)
                final_df = pd.concat([existing_annotations, updated_annotations_df], ignore_index=True)
            else:
                final_df = pd.DataFrame(updated_annotations)

            final_df.to_csv(annotation_csv_path, index=False)
            st.success(f"Annotations saved to {annotation_csv_path}")
            st.session_state.annotations = []  # Clear the annotations after exporting
            

# Example usage
output_directory = "/home/uril/hot3d/hot3d/dataset/Labelled/Videos"
annotation_csv_path = "/home/uril/hot3d/hot3d/dataset/Labelled/Videos/new_annotations.csv"  # Path to the pre-existing annotations file
annotate_video(output_directory, annotation_csv_path)

