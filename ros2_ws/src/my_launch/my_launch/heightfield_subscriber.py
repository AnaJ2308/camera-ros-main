#!/usr/bin/env python3
# subscribes to the 3 pointcloud topics and creates a PNG for MuJoCo to model
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformListener
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

import numpy as np
from collections import deque
from PIL import Image
import os
from typing import Dict

class HeightfieldBuilder(Node):
    """
    Fuses 3 PointCloud2 topics, transforms them into target_frame (Z-up),
    crops to an XY square, downsamples, rasterizes to a grid, and writes a 16-bit PNG.
    """

    def __init__(self):
        super().__init__('heightfield_subscriber')

        # ---------- Parameters (declare + get so you can override with ROS params) ----------
        self.declare_parameter('target_frame', 'base_link')     # your static Z-up frame
        self.declare_parameter('topics', [
            '/topic_1/cam_1/depth/color/points',   # <-- set these to your actual topics
            '/topic_2/cam_2/depth/color/points',
            '/topic_3/cam_3/depth/color/points'
        ])
        self.declare_parameter('half_extent_m', 2.5)            # 2.5 -> 5x5 m; set 1.5 for 3x3 m, 2.0 for 4x4 m
        self.declare_parameter('nrow', 256)
        self.declare_parameter('ncol', 256)
        self.declare_parameter('voxel_size_m', 0.0)             # 0 -> auto = approx cell size
        self.declare_parameter('timer_hz', 2.0)                 # rasterize/export rate
        self.declare_parameter('png_path', '/tmp/mj_hfield.png')
        self.declare_parameter('aggregate', 'mean')           # 'switched median to mean' or 'max'
        self.declare_parameter('clip_percentiles', [1.0, 99.0]) # robust z-clip; set [] to disable
        self.declare_parameter('flip_x', False)   # mirror left/right (columns)
        self.declare_parameter('flip_y', True)

        self.flip_x = bool(self.get_parameter('flip_x').value)
        self.flip_y = bool(self.get_parameter('flip_y').value)

        self.target_frame: str = self.get_parameter('target_frame').get_parameter_value().string_value
        self.topics: list[str] = [str(s) for s in self.get_parameter('topics').get_parameter_value().string_array_value]
        self.half_extent = float(self.get_parameter('half_extent_m').value)
        self.nrow = int(self.get_parameter('nrow').value)
        self.ncol = int(self.get_parameter('ncol').value)
        self.timer_hz = float(self.get_parameter('timer_hz').value)
        self.png_path = str(self.get_parameter('png_path').value)
        self.aggregate = str(self.get_parameter('aggregate').value).lower()
        self.clip_pcts = [float(x) for x in self.get_parameter('clip_percentiles').value] if \
            isinstance(self.get_parameter('clip_percentiles').value, (list, tuple)) else [1.0, 99.0]

        voxel_param = float(self.get_parameter('voxel_size_m').value)
        self.cell_size = (2.0 * self.half_extent) / max(self.nrow, self.ncol)
        self.voxel_size = voxel_param if voxel_param > 0 else self.cell_size  # good default

        # ---------- TF setup ----------
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------- Subscriptions (keep only latest msg per topic) ----------
        self.latest: Dict[str, PointCloud2] = {}
        self.subs = []
        for t in self.topics:
            self.subs.append(self.create_subscription(PointCloud2, t, self._make_cb(t), 10))

        # ---------- Timer ----------
        period = 1.0 / max(self.timer_hz, 0.1)
        self.timer = self.create_timer(period, self.process)

        self.get_logger().info(
            f'HeightfieldBuilder started:\n'
            f'- target_frame: {self.target_frame}\n'
            f'- topics: {self.topics}\n'
            f'- area: {2*self.half_extent:.2f} m × {2*self.half_extent:.2f} m\n'
            f'- grid: {self.nrow}×{self.ncol}  (cell ≈ {self.cell_size*100:.1f} cm)\n'
            f'- voxel_size: {self.voxel_size*100:.1f} cm\n'
            f'- writing PNG to: {self.png_path}'
        )

    def _make_cb(self, topic_name):
        def _cb(msg: PointCloud2):
            self.latest[topic_name] = msg
        return _cb

    # ---- utility: transform PointCloud2 to target frame and convert to Nx3 numpy ----
    def cloud_to_xyz_in_target(self, msg: PointCloud2) -> np.ndarray:
        if msg.header.frame_id != self.target_frame:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.target_frame, msg.header.frame_id, rclpy.time.Time())
                msg = do_transform_cloud(msg, tf)
            except Exception as e:
                self.get_logger().warn(f'TF transform {msg.header.frame_id}→{self.target_frame} failed: {e}')
                return np.empty((0, 3), dtype=np.float32)

        # Convert to numpy (x,y,z) skipping NaNs
        rows = list(point_cloud2.read_points(msg, field_names=('x','y','z'), skip_nans=True))
        if not rows:
            return np.empty((0, 3), dtype=np.float32)

        arr = np.asarray(rows)
    # If it's a structured array (named fields), stack fields; else cast directly
        if getattr(arr, 'dtype', None) is not None and arr.dtype.names:
            xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float32, copy=False)
        else:
            xyz = arr.astype(np.float32, copy=False).reshape(-1, 3)
        return xyz

    # ---- utility: voxel downsample in XYZ via quantization ----
    def voxel_downsample(self, xyz: np.ndarray, voxel: float) -> np.ndarray:
        if xyz.shape[0] == 0 or voxel <= 0:
            return xyz
        q = np.floor(xyz / voxel).astype(np.int64)
        # unique rows on quantized indices, keep first occurrence
        _, keep = np.unique(q, axis=0, return_index=True)
        return xyz[np.sort(keep)]

    def process(self):
        # Ensure we have at least one cloud from each topic
        if not all(t in self.latest for t in self.topics):
            return

        # 1) Transform all clouds into target_frame and concatenate
        clouds = []
        for t in self.topics:
            xyz = self.cloud_to_xyz_in_target(self.latest[t])
            if xyz.size:
                clouds.append(xyz)
        if not clouds:
            return
        xyz = np.concatenate(clouds, axis=0)

        # 2) Crop to square ROI centered at (0,0) in target frame
        L = self.half_extent
        m = (xyz[:,0] >= -L) & (xyz[:,0] <= L) & (xyz[:,1] >= -L) & (xyz[:,1] <= L)
        xyz = xyz[m]
        if xyz.shape[0] == 0:
            self.get_logger().warn('No points in ROI; skipping.')
            return

        # 3) Optional robust z clipping to drop outliers
        if len(self.clip_pcts) == 2:
            lo, hi = np.percentile(xyz[:,2], self.clip_pcts)
            xyz[:,2] = np.clip(xyz[:,2], lo, hi)

        # 4) Voxel downsample (≈ cell size)
        xyz = self.voxel_downsample(xyz, self.voxel_size)

        # 5) Rasterize to grid (aggregate Z per cell)
        # map XY to [0..ncol-1], [0..nrow-1]
        gx = ((xyz[:,0] + L) / (2*L) * (self.ncol - 1)).astype(np.int32)
        gy = ((xyz[:,1] + L) / (2*L) * (self.nrow - 1)).astype(np.int32)
        z  = xyz[:,2].astype(np.float32)

        H = np.full((self.nrow, self.ncol), np.nan, dtype=np.float32)
        flat_idx = gy * self.ncol + gx

        if self.aggregate == 'max':
            Hflat = np.full(self.nrow*self.ncol, -np.inf, dtype=np.float32)
            np.maximum.at(Hflat, flat_idx, z)
            H = Hflat.reshape(self.nrow, self.ncol)
            H[H == -np.inf] = np.nan
        elif self.aggregate == 'mean':
            counts = np.bincount(flat_idx, minlength=self.nrow*self.ncol)
            sums   = np.bincount(flat_idx, weights=z, minlength=self.nrow*self.ncol)
            mean_flat = np.divide(
                sums, counts,
                out=np.full(self.nrow*self.ncol, np.nan, dtype=np.float32),
                where=counts > 0
            )
            H = mean_flat.reshape(self.nrow, self.ncol)
        else:  # median (robust)
            from collections import defaultdict
            buckets = defaultdict(list)
            for idx, zz in zip(flat_idx, z):
                buckets[idx].append(zz)
            for idx, vals in buckets.items():
                i = idx // self.ncol
                j = idx %  self.ncol
                H[i, j] = float(np.median(vals))

        # 6) Fill small holes with simple neighbor averaging (3 passes)
        mask = np.isnan(H)
        if np.any(mask):
            for _ in range(3):
                Hp = np.pad(H, 1, mode='edge')
                neigh = np.stack([
                    Hp[0:-2,0:-2], Hp[0:-2,1:-1], Hp[0:-2,2:  ],
                    Hp[1:-1,0:-2],                 Hp[1:-1,2:  ],
                    Hp[2:  ,0:-2], Hp[2:  ,1:-1], Hp[2:  ,2:  ],
                ], axis=0)
                with np.errstate(invalid='ignore'):
                    mean = np.nanmean(neigh, axis=0)
                fill = np.isnan(H) & ~np.isnan(mean)
                H[fill] = mean[fill]
                if not np.any(np.isnan(H)):
                    break
            if np.any(np.isnan(H)):
                H[np.isnan(H)] = np.nanmin(H)
        
        if self.flip_y:
            H = np.flipud(H)   # flip along Y (rows)
        if self.flip_x:
            H = np.fliplr(H)   # flip along X (cols)

        # 7) Normalize to 0..1 and write 16-bit PNG
        zmin = float(np.nanmin(H))
        zmax = float(np.nanmax(H))
        if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
            self.get_logger().warn('Degenerate heightfield; skipping write.')
            return

        H01 = (H - zmin) / (zmax - zmin)
        img16 = (H01 * 65535.0).astype(np.uint16)
        os.makedirs(os.path.dirname(self.png_path), exist_ok=True)
        Image.fromarray(img16).save(self.png_path)

        self.get_logger().info(
            f'PNG written: {self.png_path} | '
            f'area={2*L:.2f}m | grid={self.nrow}x{self.ncol} | '
            f'cell≈{self.cell_size*100:.1f}cm | '
            f'Z[{zmin:.3f},{zmax:.3f}]'
        )

def main():
    rclpy.init()
    rclpy.spin(HeightfieldBuilder())
    rclpy.shutdown()

if __name__ == '__main__':
    main()