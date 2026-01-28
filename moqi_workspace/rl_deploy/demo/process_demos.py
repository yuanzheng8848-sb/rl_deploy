import os
import pickle as pkl
import numpy as np
from pathlib import Path
import shutil
import cv2
from scipy.spatial.transform import Rotation as R

def process_demos(source_dir, dest_file):
    """
    Reads .pkl files from source_dir, filters out idle frames,
    merges them, and saves to a single dest_file.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_file)
    
    if not source_path.exists():
        print(f"Source directory {source_path} does not exist.")
        return

    print(f"Processing demos from {source_path} to {dest_path}")

    # Thresholds for "idle" detection
    TRANS_THRESH = 0.001  # 1mm
    ROT_THRESH = 0.01     # ~0.57 degrees
    GRIPPER_THRESH = 0.01 # Gripper command change threshold

    files = list(source_path.glob("*.pkl"))
    files.sort() # Ensure deterministic order
    print(f"Found {len(files)} files.")

    total_original_frames = 0
    total_kept_frames = 0
    
    all_transitions = []

    for file_path in files:
        try:
            with open(file_path, "rb") as f:
                transitions = pkl.load(f)
            
            if not transitions:
                print(f"Skipping empty file: {file_path.name}")
                continue

            original_len = len(transitions)
            kept_transitions = []
            
            # Get Reset Pose from the VERY first frame (before filtering)
            reset_t = transitions[0]
            reset_state_vec = reset_t['observations']['state'] # [x,y,z,qx,qy,qz,qw, grip]
            reset_pos = reset_state_vec[:3]
            reset_quat = reset_state_vec[3:7]
            
            # Construct T_reset
            r_reset = R.from_quat(reset_quat)
            T_reset = np.eye(4)
            T_reset[:3, :3] = r_reset.as_matrix()
            T_reset[:3, 3] = reset_pos
            T_reset_inv = np.linalg.inv(T_reset)

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
                obs_quat = obs_state[3:7] # [qx,qy,qz,qw]
                obs_grip = obs_state[7:8] # Gripper (1,)
                
                # Parse Action
                action = t['actions']
                delta_pos_body = action[:3]
                gripper_cmd = action[6]

                # Reconstruct Target Pos (World) for change detection
                r_curr = R.from_quat(obs_quat)
                target_pos = obs_pos + r_curr.apply(delta_pos_body)
                
                # Check for changes (Idle filtering)
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
            
            # Post-processing: Resize images and Compute Relative State
            
            file_transitions = []
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
                
                # 2. Compute Relative State (13-dim)
                # Input State: [x,y,z,qx,qy,qz,qw, grip] (8 dim)
                
                def compute_13dim_state(raw_state_8dim, T_inv):
                    # Extract Absolute
                    p_abs = raw_state_8dim[:3]
                    q_abs = raw_state_8dim[3:7]
                    g_val = raw_state_8dim[7:8]
                    
                    # Compute Relative T
                    r_curr = R.from_quat(q_abs)
                    T_curr = np.eye(4)
                    T_curr[:3, :3] = r_curr.as_matrix()
                    T_curr[:3, 3] = p_abs
                    
                    T_rel = T_inv @ T_curr
                    
                    # Extract Relative Pos + Euler
                    p_rel = T_rel[:3, 3]
                    r_rel = R.from_matrix(T_rel[:3, :3])
                    e_rel = r_rel.as_euler('xyz')
                    
                    # Dummy Vel
                    vel = np.zeros(6)
                    
                    # Pack: [Gripper(1), Pos(3), Euler(3), Vel(6)] = 13
                    # Wait, Agent code: np.concatenate([curr_gripper, curr_pose_euler(6), vel(6)])
                    # And curr_pose_euler was [xyz, rpy]
                    # So: [Gripper(1), X, Y, Z, R, P, Y, Vx...Vz]
                    
                    return np.concatenate([g_val, p_rel, e_rel, vel])

                if 'state' in new_obs:
                     new_obs['state'] = compute_13dim_state(new_obs['state'], T_reset_inv)
                
                if new_t['next_observations'] is not None and 'state' in new_t['next_observations']:
                     new_t['next_observations']['state'] = compute_13dim_state(new_t['next_observations']['state'], T_reset_inv)
                
                new_t['observations'] = new_obs
                file_transitions.append(new_t)

            # Append to main list
            new_len = len(file_transitions)
            total_original_frames += original_len
            total_kept_frames += new_len
            all_transitions.extend(file_transitions)
            
            print(f"Processed {file_path.name}: {original_len} -> {new_len} frames ({new_len/original_len:.1%})")

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    # Save merged file
    print("-" * 40)
    print(f"Total Frames: {total_original_frames} -> {total_kept_frames}")
    if total_original_frames > 0:
        print(f"Overall Reduction: {total_kept_frames/total_original_frames:.1%}")
    
    with open(dest_path, "wb") as f:
        pkl.dump(all_transitions, f)
    
    print(f"Merged demos saved to {dest_path}")

if __name__ == "__main__":
    # Define paths relative to this script
    current_dir = Path(__file__).parent
    # Source: demos_origin in current directory
    source_dir = current_dir / "demos_origin"
    # Dest: merged_demos.pkl in current directory
    dest_file = current_dir / "merged_demos.pkl"
    
    process_demos(source_dir, dest_file)
