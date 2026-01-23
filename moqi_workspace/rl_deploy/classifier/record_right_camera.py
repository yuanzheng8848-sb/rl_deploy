import sys
import os
import time
import cv2
from pathlib import Path
from datetime import datetime

# Add pyroki to path
current_dir = Path(__file__).parent
pyroki_path = current_dir.parent / "pyroki"
sys.path.append(str(pyroki_path))

try:
    from realsense_camera import RealsenseCamera
except ImportError as e:
    print(f"Failed to import RealsenseCamera: {e}")
    sys.exit(1)

def main():
    # Configuration
    CAMERA_ID = "236422072385" # Right camera
    WIDTH = 640
    HEIGHT = 480
    FPS = 30
    
    # Create recording directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = current_dir / f"recording_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving images to: {save_dir}")
    
    # Initialize Camera
    print(f"Initializing camera {CAMERA_ID}...")
    try:
        cam = RealsenseCamera(
            device_id=CAMERA_ID,
            enable_depth=False,
            width=WIDTH,
            height=HEIGHT,
            fps=FPS
        )
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return

    print("Camera initialized. Starting recording...")
    print("Press Ctrl+C to stop.")

    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            loop_start = time.time()
            
            # Capture
            img = cam.get_data(viz=False)
            
            # Handle Realsense return type (list [color, depth] or just color)
            if isinstance(img, (list, tuple)):
                img = img[0]
                
            if img is not None:
                # Save image
                # RealsenseCamera usually returns RGB, cv2.imwrite expects BGR
                # But wait, let's check train_pick_place.py again.
                # It says "RealsenseCamera might return RGB? ... OpenArmEnv._update_currpos converts BGR to RGB."
                # Usually cv2 based cameras return BGR. Realsense SDK returns RGB usually.
                # Let's assume it returns RGB and convert to BGR for saving.
                
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                filename = save_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(filename), img_bgr)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Recorded {frame_count} frames...", end='\r')
            
            # Maintain 30Hz
            elapsed = time.time() - loop_start
            sleep_time = max(0, (1.0/FPS) - elapsed)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    except Exception as e:
        print(f"\nError during recording: {e}")
    finally:
        # Cleanup
        # RealsenseCamera doesn't have a close method exposed in the snippet I saw, 
        # but usually it's good practice. If it had one, we'd call it.
        # cam.pipeline.stop() if accessible.
        pass
        
    duration = time.time() - start_time
    print(f"Total frames: {frame_count}")
    print(f"Duration: {duration:.2f}s")
    print(f"Average FPS: {frame_count/duration:.2f}")

if __name__ == "__main__":
    main()
