import cv2
import os
import glob

from config import OUTPUT_PATH, VIDEO_OUTPUT, OUTPUT_FPS


def create_video():
    frames = sorted(glob.glob(os.path.join(OUTPUT_PATH, "*.png")))
    
    if not frames:
        print("No frames found in output_frames")
        return
    
    first_frame = cv2.imread(frames[0])
    height, width = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, OUTPUT_FPS, (width, height))
    
    for frame_path in frames:
        frame = cv2.imread(frame_path)
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Video saved to: {VIDEO_OUTPUT}")

if __name__ == "__main__":
    create_video()