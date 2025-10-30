import cv2
import numpy as np
from collections import deque

class VideoStabilizer:
    def __init__(self, smoothing_radius=30):
        self.prev_gray = None
        self.prev_pts = None
        self.transforms = []
        self.smoothing_radius = smoothing_radius
        self.trajectory = np.zeros(3)  # cumulative [dx, dy, da]
        self.smoothed_trajectory = np.zeros(3)
        self.trajectory_buffer = deque(maxlen=smoothing_radius)

    def stabilize_frame(self, frame):
        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            # First frame, just initialize and return it as is
            self.prev_gray = frame_gray
            return frame

        # Detect features in previous frame if not already available
        if self.prev_pts is None:
            self.prev_pts = cv2.goodFeaturesToTrack(
                self.prev_gray, maxCorners=200, qualityLevel=0.01,
                minDistance=30, blockSize=3
            )

        if self.prev_pts is None:
            # If no features, just pass the frame
            self.prev_gray = frame_gray
            return frame

        # Calculate optical flow (track feature points)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, frame_gray, self.prev_pts, None
        )

        # Keep valid points
        idx = np.where(status == 1)[0]
        prev_pts, curr_pts = self.prev_pts[idx], curr_pts[idx]

        if len(prev_pts) < 4 or len(curr_pts) < 4:
            # Not enough points to estimate transform
            self.prev_gray = frame_gray
            self.prev_pts = None
            return frame

        # Estimate affine transform
        m, _ = cv2.estimateAffine2D(prev_pts, curr_pts)

        if m is None:
            # Fall back if transform can't be estimated
            self.prev_gray = frame_gray
            self.prev_pts = None
            return frame

        dx, dy = m[0, 2], m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        # Update trajectory
        self.trajectory += [dx, dy, da]
        self.trajectory_buffer.append(self.trajectory.copy())

        # Smooth trajectory using moving average
        self.smoothed_trajectory = np.mean(self.trajectory_buffer, axis=0)

        # Compute difference and apply to transforms
        diff = self.smoothed_trajectory - self.trajectory
        dx += diff[0]
        dy += diff[1]
        da += diff[2]

        # Recreate transformation matrix with smoothed values
        m_smooth = np.array([
            [np.cos(da), -np.sin(da), dx],
            [np.sin(da),  np.cos(da), dy]
        ])

        # Apply the affine transform to stabilize
        h, w = frame.shape[:2]
        stabilized_frame = cv2.warpAffine(frame, m_smooth, (w, h))

        # Update for next iteration
        self.prev_gray = frame_gray
        self.prev_pts = curr_pts.reshape(-1, 1, 2)

        return stabilized_frame
 