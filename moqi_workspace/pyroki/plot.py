import numpy as np
import csv
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

class PoseRecorderMatplotlib:
    """
    Pose recorder + optimized Matplotlib 3D visualization for large datasets.
    Input pose per sample: 1D length-7 array [qx, qy, qz, qw, x, y, z]
    """

    def __init__(self):
        self._records = []

    @staticmethod
    def _validate_7(pose, name="pose"):
        a = np.asarray(pose, dtype=float)
        if a.ndim != 1 or a.size != 7:
            raise ValueError(f"{name} must be 1D length-7: [qx,qy,qz,qw,x,y,z]. Got {a.shape}")
        return a.copy()

    def append(self, timestamp, target_pose7, actual_pose7):
        ts = float(timestamp)
        tgt = self._validate_7(target_pose7, "target_pose")
        act = self._validate_7(actual_pose7, "actual_pose")
        self._records.append((ts, tgt, act))

    def to_numpy(self):
        if not self._records:
            return np.array([]), np.zeros((0,7)), np.zeros((0,7))
        t = np.array([r[0] for r in self._records], dtype=float)
        targets = np.stack([r[1] for r in self._records], axis=0)
        actuals = np.stack([r[2] for r in self._records], axis=0)
        return t, targets, actuals

    def save_csv(self, filename):
        t, targets, actuals = self.to_numpy()
        with open(filename, 'w', newline='') as f:
            w = csv.writer(f)
            header = ['timestamp',
                      'target_qx','target_qy','target_qz','target_qw','target_x','target_y','target_z',
                      'actual_qx','actual_qy','actual_qz','actual_qw','actual_x','actual_y','actual_z']
            w.writerow(header)
            for i, tt in enumerate(t):
                tgt = targets[i]; act = actuals[i]
                row = [float(tt)] + tgt.tolist() + act.tolist()
                w.writerow(row)

    # ----------------- optimized plotting helpers -----------------
    @staticmethod
    def _sampled_indices(length, max_items):
        if length <= max_items:
            return np.arange(length)
        step = int(np.ceil(length / max_items))
        return np.arange(0, length, step)

    def _frame_indices(self, idx_array, ratio):
        if len(idx_array) == 0:
            return np.array([], dtype=int)
        k = max(1, int(len(idx_array) * ratio))
        if k >= len(idx_array):
            return idx_array
        step = int(np.ceil(len(idx_array) / k))
        return idx_array[::step]

    def _axes_at_indices(self, quats, indices):
        if len(indices) == 0:
            return np.zeros((0,3)), np.zeros((0,3)), np.zeros((0,3))
        rots = R.from_quat(quats[indices])  # quat order [qx,qy,qz,qw]
        mats = rots.as_matrix()
        vx = mats[:, :, 0]
        vy = mats[:, :, 1]
        vz = mats[:, :, 2]
        return vx, vy, vz

    # ------------------ main optimized plotting ------------------
    def plot_3d(self,
                frame_scale=None,
                max_markers=200,
                frame_marker_ratio=1.0,
                line_step=1,
                show_frames_on='actual',
                show_connect_lines=False,
                draw_target=True,
                draw_actual=True,
                ignore_initial_seconds=10.0,
                align_to_target_start=True,
                figsize=(10,8)):
        """
        Optimized plotting for large trajectories.

        Parameters:
          - frame_scale: arrow length for local frames; if None will auto-scale ~2% of bbox diagonal
          - max_markers: maximum number of sampled markers (per trajectory) to draw
          - frame_marker_ratio: fraction of markers that will have frames drawn (0..1)
          - line_step: initial downsampling for the trajectory line (auto-increased if many points)
          - show_frames_on: 'actual'|'target'|'both' where to draw local frames
          - show_connect_lines: draw thin lines between sampled corresponding points to show error
          - draw_target/draw_actual: whether to draw target/actual trajectories
          - ignore_initial_seconds: ignore records whose timestamp < (t0 + ignore_initial_seconds)
          - align_to_target_start: if True, translate both trajectories so target's first position becomes origin (only for plotting)
        """
        t, targets, actuals = self.to_numpy()
        if t.size == 0:
            print("No data to plot.")
            return

        # ---- trim initial seconds to avoid init transients ----
        t0 = t[0]
        if ignore_initial_seconds is not None and ignore_initial_seconds > 0:
            cutoff = t0 + float(ignore_initial_seconds)
            mask = t >= cutoff
            if not np.any(mask):
                print(f"No samples after ignore_initial_seconds={ignore_initial_seconds}s; plotting nothing.")
                return
            t = t[mask]
            targets = targets[mask]
            actuals = actuals[mask]

        pos_t = targets[:, 4:7] if targets.size else np.zeros((0,3))
        quat_t = targets[:, 0:4] if targets.size else np.zeros((0,4))
        pos_a = actuals[:, 4:7] if actuals.size else np.zeros((0,3))
        quat_a = actuals[:, 0:4] if actuals.size else np.zeros((0,4))

        N_t = pos_t.shape[0]
        N_a = pos_a.shape[0]
        N = max(N_t, N_a)

        # ---------------- auto-tune for large N ----------------
        if N > 5000:
            max_markers = min(max_markers * 3, 1000)
            line_step = max(line_step, int(np.ceil(N / 5000)))
        max_markers = max(10, int(max_markers))
        line_step = max(1, int(line_step))

        # compute bounding box to auto-scale arrow length
        all_pos = np.vstack([pos_t, pos_a]) if (N_t > 0 and N_a > 0) else (pos_t if N_t>0 else pos_a)
        bbox = all_pos.max(axis=0) - all_pos.min(axis=0)
        diag = np.linalg.norm(bbox)
        if frame_scale is None:
            frame_scale = max(diag * 0.02, 1e-4)

        # ---------------- alignment: translate so target start = origin (plot-only) ----------------
        if align_to_target_start and N_t > 0:
            origin = pos_t[0].copy()
            pos_t_plot = pos_t - origin
            pos_a_plot = pos_a - origin
        else:
            pos_t_plot = pos_t.copy()
            pos_a_plot = pos_a.copy()

        # sampled marker indices
        idx_t = self._sampled_indices(N_t, max_markers)
        idx_a = self._sampled_indices(N_a, max_markers)

        # frame indices (subset of sampled indices)
        frames_t_idx = self._frame_indices(idx_t, frame_marker_ratio)
        frames_a_idx = self._frame_indices(idx_a, frame_marker_ratio)

        # compute axis vectors for frames using original quaternions (rotation unaffected by translation)
        vx_t, vy_t, vz_t = self._axes_at_indices(quat_t, frames_t_idx)
        vx_a, vy_a, vz_a = self._axes_at_indices(quat_a, frames_a_idx)

        # ------------------ plotting ------------------
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect((1,1,1))

        # trajectory lines (downsample by line_step), using aligned positions for plotting
        if draw_target and N_t > 0:
            ax.plot(pos_t_plot[::line_step,0], pos_t_plot[::line_step,1], pos_t_plot[::line_step,2],
                    color='orange', linestyle='--', linewidth=1.2, alpha=0.9, label='target')
            if len(idx_t) > 0:
                ax.scatter(pos_t_plot[idx_t,0], pos_t_plot[idx_t,1], pos_t_plot[idx_t,2],
                           color='orange', s=14, marker='o', alpha=0.9)

        if draw_actual and N_a > 0:
            ax.plot(pos_a_plot[::line_step,0], pos_a_plot[::line_step,1], pos_a_plot[::line_step,2],
                    color='blue', linestyle='-', linewidth=1.2, alpha=0.9, label='actual')
            if len(idx_a) > 0:
                ax.scatter(pos_a_plot[idx_a,0], pos_a_plot[idx_a,1], pos_a_plot[idx_a,2],
                           color='blue', s=8, marker='.', alpha=0.9)

        # start / end markers (aligned)
        if N_t > 0:
            ax.scatter(pos_t_plot[0,0], pos_t_plot[0,1], pos_t_plot[0,2], color='orange', s=60, marker='D', label='t_start')
            ax.scatter(pos_t_plot[-1,0], pos_t_plot[-1,1], pos_t_plot[-1,2], color='orange', s=60, marker='X', label='t_end')
        if N_a > 0:
            ax.scatter(pos_a_plot[0,0], pos_a_plot[0,1], pos_a_plot[0,2], color='blue', s=60, marker='D', label='a_start')
            ax.scatter(pos_a_plot[-1,0], pos_a_plot[-1,1], pos_a_plot[-1,2], color='blue', s=60, marker='X', label='a_end')

        # batch quiver for frames (X:red, Y:green, Z:blue)
        # Note: quiver uses direction vectors only; translations already applied to pos_*_plot
        if show_frames_on in ('target','both') and len(frames_t_idx) > 0:
            pts = pos_t_plot[frames_t_idx]
            ax.quiver(pts[:,0], pts[:,1], pts[:,2],
                      vx_t[:,0], vx_t[:,1], vx_t[:,2],
                      length=frame_scale, normalize=True, color='r', linewidth=0.5)
            ax.quiver(pts[:,0], pts[:,1], pts[:,2],
                      vy_t[:,0], vy_t[:,1], vy_t[:,2],
                      length=frame_scale, normalize=True, color='g', linewidth=0.5)
            ax.quiver(pts[:,0], pts[:,1], pts[:,2],
                      vz_t[:,0], vz_t[:,1], vz_t[:,2],
                      length=frame_scale, normalize=True, color='b', linewidth=0.5)

        if show_frames_on in ('actual','both') and len(frames_a_idx) > 0:
            pts = pos_a_plot[frames_a_idx]
            ax.quiver(pts[:,0], pts[:,1], pts[:,2],
                      vx_a[:,0], vx_a[:,1], vx_a[:,2],
                      length=frame_scale, normalize=True, color='r', linewidth=0.5)
            ax.quiver(pts[:,0], pts[:,1], pts[:,2],
                      vy_a[:,0], vy_a[:,1], vy_a[:,2],
                      length=frame_scale, normalize=True, color='g', linewidth=0.5)
            ax.quiver(pts[:,0], pts[:,1], pts[:,2],
                      vz_a[:,0], vz_a[:,1], vz_a[:,2],
                      length=frame_scale, normalize=True, color='b', linewidth=0.5)

        # optional connecting lines between sampled corresponding points to show error (using aligned coords)
        if show_connect_lines:
            L = min(N_t, N_a)
            if L > 0:
                conn_idx = self._sampled_indices(L, max_markers)
                for i in conn_idx:
                    p1 = pos_t_plot[min(i, N_t-1)]
                    p2 = pos_a_plot[min(i, N_a-1)]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                            color='gray', linewidth=0.6, alpha=0.5)

        # labels, legend, limits
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend(loc='upper left')

        # auto limits with margin (based on aligned positions)
        all_pos_plot = np.vstack([pos_t_plot, pos_a_plot]) if (N_t > 0 and N_a > 0) else (pos_t_plot if N_t>0 else pos_a_plot)
        min_xyz = all_pos_plot.min(axis=0)
        max_xyz = all_pos_plot.max(axis=0)
        margin = np.maximum((max_xyz - min_xyz) * 0.05, 0.01)
        ax.set_xlim(min_xyz[0]-margin[0], max_xyz[0]+margin[0])
        ax.set_ylim(min_xyz[1]-margin[1], max_xyz[1]+margin[1])
        ax.set_zlim(min_xyz[2]-margin[2], max_xyz[2]+margin[2])

        plt.title('Target vs Actual (aligned to target start)')
        plt.tight_layout()
        plt.show()