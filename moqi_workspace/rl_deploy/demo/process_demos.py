import os
import pickle as pkl
import numpy as np
from pathlib import Path
import shutil
import cv2

def process_demos(source_dir, dest_dir):
    """
    Reads .pkl files from source_dir, filters out idle frames,
    and saves them to dest_dir.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_path} does not exist.")
        return

    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    print(f"Processing demos from {source_path} to {dest_path}")

    # Thresholds for "idle" detection
    TRANS_THRESH = 0.001  # 1mm
    ROT_THRESH = 0.01     # ~0.57 degrees
    GRIPPER_THRESH = 0.01 # Gripper command change threshold

    files = list(source_path.glob("*.pkl"))
    print(f"Found {len(files)} files.")

    total_original_frames = 0
    total_kept_frames = 0

    for file_path in files:
        try:
            with open(file_path, "rb") as f:
                transitions = pkl.load(f)
            
            if not transitions:
                print(f"Skipping empty file: {file_path.name}")
                continue

            original_len = len(transitions)
            kept_transitions = []
            
            # Always keep the last 10 frames
            # But we process from the start to filter the beginning/middle
            
            from scipy.spatial.transform import Rotation as R

            prev_obs_pose = None
            prev_target_pos = None
            prev_gripper_cmd = None
            
            for i, t in enumerate(transitions):
                # Always keep last 10 frames
                if i >= original_len - 10:
                    kept_transitions.append(t)
                    continue

                # Parse Obs
                obs_state = t['observations']['state'] # [x,y,z,qx,qy,qz,qw, grip]
                obs_pos = obs_state[:3]
                obs_quat = obs_state[3:7] # [qx,qy,qz,qw] - check order in record_demo_rl.py
                # record_demo_rl.py: obs_right_ee is [x,y,z,qx,qy,qz,qw]
                # scipy expects [x,y,z,w] usually, but let's check.
                # record_demo_rl.py: to_xyzquat puts w last -> [x,y,z,w]
                # Wait, record_demo_rl.py line 659: np.concatenate([p[4:], p[1:4], [p[0]]])
                # p is [qw, qx, qy, qz, x, y, z]
                # p[4:] is [x,y,z]
                # p[1:4] is [qx,qy,qz]
                # p[0] is qw
                # So obs_state is [x,y,z, qx,qy,qz,qw].
                # scipy R.from_quat expects [x,y,z,w]. So this matches.
                
                # Parse Action
                action = t['actions']
                delta_pos_body = action[:3]
                delta_euler = action[3:6]
                gripper_cmd = action[6]

                # Reconstruct Target Pos (World)
                r_curr = R.from_quat(obs_quat)
                target_pos = obs_pos + r_curr.apply(delta_pos_body)
                
                # Check for changes
                is_active = False
                
                if prev_obs_pose is not None:
                    # Check if Robot Moved
                    obs_move = np.linalg.norm(obs_pos - prev_obs_pose[:3])
                    if obs_move > TRANS_THRESH:
                        is_active = True
                    
                    # Check if Target Moved (Human Input Changed)
                    target_move = np.linalg.norm(target_pos - prev_target_pos)
                    if target_move > TRANS_THRESH:
                        is_active = True
                        
                    # Check Gripper
                    if abs(gripper_cmd - prev_gripper_cmd) > GRIPPER_THRESH:
                        is_active = True
                else:
                    # First frame, keep it
                    is_active = True

                if is_active:
                    kept_transitions.append(t)
                    prev_obs_pose = obs_state
                    prev_target_pos = target_pos
                    prev_gripper_cmd = gripper_cmd
                
                # Debug stats for first few frames
                # if i < 5:
                #     print(f"Frame {i}: Active={is_active}")
            
            # Calculate stats for this file
            print(f"  Stats: Kept {len(kept_transitions)}/{original_len}")

            # Post-processing: Resize images and Pad state
            import cv2
            
            final_transitions = []
            for t in kept_transitions:
                new_t = t.copy()
                
                # 1. Resize Images
                # Observations
                new_obs = new_t['observations'].copy()
                for k, v in new_obs.items():
                    if k.startswith('image_') and isinstance(v, np.ndarray) and v.ndim == 3:
                        if v.shape[:2] != (128, 128):
                            new_obs[k] = cv2.resize(v, (128, 128))
                
                # Next Observations (if present)
                if new_t['next_observations'] is not None:
                    new_next_obs = new_t['next_observations'].copy()
                    for k, v in new_next_obs.items():
                        if k.startswith('image_') and isinstance(v, np.ndarray) and v.ndim == 3:
                            if v.shape[:2] != (128, 128):
                                new_next_obs[k] = cv2.resize(v, (128, 128))
                    new_t['next_observations'] = new_next_obs
                
                # 2. Pad State (8 -> 14)
                # Current state: [Pose(7), Gripper(1)] -> [Pose(7), Vel(6), Gripper(1)]
                # Insert 6 zeros at index 7
                
                def pad_state(state_arr):
                    if state_arr.shape == (8,):
                        return np.insert(state_arr, 7, np.zeros(6))
                    return state_arr

                if 'state' in new_obs:
                    new_obs['state'] = pad_state(new_obs['state'])
                
                if new_t['next_observations'] is not None and 'state' in new_t['next_observations']:
                    new_t['next_observations']['state'] = pad_state(new_t['next_observations']['state'])
                
                new_t['observations'] = new_obs
                final_transitions.append(new_t)

            # Save processed file
            new_len = len(final_transitions)
            total_original_frames += original_len
            total_kept_frames += new_len
            
            dest_file_path = dest_path / file_path.name
            with open(dest_file_path, "wb") as f:
                pkl.dump(final_transitions, f)
            
            print(f"Processed {file_path.name}: {original_len} -> {new_len} frames ({new_len/original_len:.1%})")

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    print("-" * 40)
    print(f"Total Frames: {total_original_frames} -> {total_kept_frames}")
    if total_original_frames > 0:
        print(f"Overall Reduction: {total_kept_frames/total_original_frames:.1%}")
    print(f"Processed files saved to {dest_path}")

if __name__ == "__main__":
    # Define paths relative to this script
    current_dir = Path(__file__).parent
    # moqi_workspace/rl_deploy -> moqi_workspace/bc_demos
    source_dir = current_dir.parent / "bc_demos"
    dest_dir = current_dir.parent / "bc_demos_processed"
    
    process_demos(source_dir, dest_dir)
