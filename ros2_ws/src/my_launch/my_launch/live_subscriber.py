#!/usr/bin/env python3
# my current live subscriber that doesn't give me a good pointcloud
"""
Jetson HField Server (ROS2 -> TCP streamer)

- Subscribes to THREE PointCloud2 topics on the Jetson
- Transforms each cloud to a common Z-up frame
- Crops to a square ROI, optional voxel downsample
- Rasterizes with MAX-Z per cell into an (nrow x ncol) grid
- Normalizes (0..1) and streams the grid over TCP to a laptop viewer
"""

import socket, struct, zlib, time, threading
from typing import Dict, List, Tuple, Optional
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformListener
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

MAGIC = b'HFLD'   # 4 bytes
VERSION = 1

# ---------- Utils ----------

def pc2_to_xyz_np(msg: PointCloud2) -> np.ndarray:
    it = point_cloud2.read_points(msg, field_names=('x','y','z'), skip_nans=True)
    arr = np.fromiter(it, dtype=[('x','f4'),('y','f4'),('z','f4')])
    if arr.size == 0:
        return np.empty((0,3), np.float32)
    return np.column_stack((arr['x'], arr['y'], arr['z'])).astype(np.float32, copy=False)

def voxel_downsample(xyz: np.ndarray, voxel: float) -> np.ndarray:
    if xyz.shape[0] == 0 or voxel <= 0:
        return xyz
    q = np.floor(xyz / voxel).astype(np.int64)
    _, keep = np.unique(q, axis=0, return_index=True)
    return xyz[np.sort(keep)]

def rasterize_agg(points_xyz: np.ndarray,
                  nrow: int, ncol: int,
                  bounds: tuple[float,float,float,float],
                  agg: str = 'mean',                 # 'mean' | 'max' | 'median'
                  prev_grid: np.ndarray | None = None,
                  smooth_passes: int = 1) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    if points_xyz.size == 0:
        return prev_grid if prev_grid is not None else np.zeros((nrow, ncol), np.float32)

    x = points_xyz[:, 0]; y = points_xyz[:, 1]; z = points_xyz[:, 2].astype(np.float32, copy=False)

    # Map (x,y) -> integer cells. Using (n-1) avoids biasing the last column/row.
    ix = ((x - xmin) / (xmax - xmin) * (ncol - 1)).astype(np.int32)
    iy = ((y - ymin) / (ymax - ymin) * (nrow - 1)).astype(np.int32)
    np.clip(ix, 0, ncol - 1, out=ix)
    np.clip(iy, 0, nrow - 1, out=iy)
    flat_idx = iy * ncol + ix

    if agg == 'mean':
        counts = np.bincount(flat_idx, minlength=nrow * ncol)
        sums   = np.bincount(flat_idx, weights=z, minlength=nrow * ncol)
        Hflat = np.divide(
            sums, counts,
            out=np.full(nrow * ncol, np.nan, np.float32),
            where=counts > 0
        )
        H = Hflat.reshape(nrow, ncol)

    elif agg == 'median':
        from collections import defaultdict
        buckets = defaultdict(list)
        for idx, zz in zip(flat_idx, z):
            buckets[idx].append(zz)
        H = np.full((nrow, ncol), np.nan, np.float32)
        for idx, vals in buckets.items():
            i = idx // ncol; j = idx % ncol
            H[i, j] = float(np.median(vals))

    else:  # 'max'
        Hflat = np.full(nrow * ncol, -np.inf, np.float32)
        np.maximum.at(Hflat, flat_idx, z)
        H = Hflat.reshape(nrow, ncol)
        H[H == -np.inf] = np.nan

    # Fill empty cells from previous grid (prevents holes/flicker)
    if prev_grid is not None:
        mask = np.isnan(H)
        if mask.any():
            H[mask] = prev_grid[mask]

    # Light smoothing for any remaining NaNs / speckle (0–3 passes recommended)
    for _ in range(max(smooth_passes, 0)):
        if not np.isnan(H).any():
            break
        Hp = np.pad(H, 1, mode='edge')
        neigh = np.stack([
            Hp[0:-2,1:-1], Hp[2:,1:-1], Hp[1:-1,0:-2], Hp[1:-1,2:],
            Hp[0:-2,0:-2], Hp[0:-2,2:], Hp[2:,0:-2], Hp[2:,2:]
        ], axis=0)
        with np.errstate(invalid='ignore'):
            m = np.nanmean(neigh, axis=0)
        nanmask = np.isnan(H) & ~np.isnan(m)
        H[nanmask] = m[nanmask]

    # Final fallback: any stubborn NaNs → current min
    if np.isnan(H).any():
        H[np.isnan(H)] = np.nanmin(H)

    return H.astype(np.float32, copy=False)


# ---------- ROS2 Node ----------

class MultiCloudNode(Node):
    def __init__(self):
        super().__init__('hfield_server')
        # Params
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('topics', [
            '/topic_1/cam_1/depth/color/points',
            '/topic_2/cam_2/depth/color/points',
            '/topic_3/cam_3/depth/color/points',
        ])
        self.declare_parameter('half_extent_m', 6.0)
        self.declare_parameter('nrow', 160)
        self.declare_parameter('ncol', 160)
        self.declare_parameter('voxel_size_m', 0.08)
        self.declare_parameter('clip_percentiles', [])
        self.declare_parameter('flip_x', False)
        self.declare_parameter('flip_y', True)
        self.declare_parameter('send_hz', 15.0)
        self.declare_parameter('aggregate', 'mean') 
        self.declare_parameter('smooth_passes', 1)
        self.declare_parameter('zmin_fixed', 0.2)
        self.declare_parameter('zmax_fixed', 5.0)

        

        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        self.topics = [str(s) for s in self.get_parameter('topics').get_parameter_value().string_array_value]
        self.L = float(self.get_parameter('half_extent_m').value)
        self.nrow = int(self.get_parameter('nrow').value)
        self.ncol = int(self.get_parameter('ncol').value)
        self.voxel = float(self.get_parameter('voxel_size_m').value)
        self.clip_pcts = list(self.get_parameter('clip_percentiles').value) or []
        self.flip_x = bool(self.get_parameter('flip_x').value)
        self.flip_y = bool(self.get_parameter('flip_y').value)
        self.send_hz = float(self.get_parameter('send_hz').value)
        zmin_param = self.get_parameter('zmin_fixed').value
        zmax_param = self.get_parameter('zmax_fixed').value
        self.zmin_fixed = float(zmin_param) if zmin_param is not None else None
        self.zmax_fixed = float(zmax_param) if zmax_param is not None else None

        self.aggregate = str(self.get_parameter('aggregate').value).lower()
        self.smooth_passes = int(self.get_parameter('smooth_passes').value)

        self.cell_size = (2.0*self.L)/max(self.nrow,self.ncol)
        if self.voxel <= 0:
            self.voxel = self.cell_size

        # TF + subs
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=3)
        self._latest: Dict[str, Optional[PointCloud2]] = {t: None for t in self.topics}
        self._seen_first = {t: False for t in self.topics}
        for t in self.topics:
            self.create_subscription(PointCloud2, t, self._make_cb(t), qos)

        self._grid: Optional[np.ndarray] = np.zeros((self.nrow,self.ncol), np.float32)
        self._seq = 0
        self._pts_last = 0
        self.create_timer(1.0, self._stats)
        self.get_logger().info(
            f'HField server up: ROI {2*self.L:.1f}m, grid {self.nrow}x{self.ncol}, voxel {self.voxel*100:.1f}cm')

    def _make_cb(self, topic):
        def _cb(msg: PointCloud2):
            if not self._seen_first[topic]:
                self.get_logger().info(f'First message from {topic} (frame={msg.header.frame_id})')
                self._seen_first[topic] = True
            self._latest[topic] = msg
        return _cb

    def _cloud_to_xyz_in_target(self, msg: PointCloud2) -> np.ndarray:
        if msg.header.frame_id and msg.header.frame_id != self.target_frame:
            try:
                tf = self.tf_buffer.lookup_transform(self.target_frame, msg.header.frame_id, rclpy.time.Time())
                msg = do_transform_cloud(msg, tf)
            except Exception as e:
                self.get_logger().warn(f'TF {msg.header.frame_id}->{self.target_frame} failed: {e}')
                return np.empty((0,3), np.float32)
        return pc2_to_xyz_np(msg)

    def get_concat_latest_xyz(self) -> np.ndarray:
        clouds = []
        for t in self.topics:
            m = self._latest.get(t)
            if m is None: continue
            xyz = self._cloud_to_xyz_in_target(m)
            if xyz.size: clouds.append(xyz)
        if not clouds:
            return np.empty((0,3), np.float32)
        xyz = np.concatenate(clouds, axis=0)
        L = self.L
        sel = (xyz[:,0]>=-L)&(xyz[:,0]<=L)&(xyz[:,1]>=-L)&(xyz[:,1]<=L)
        xyz = xyz[sel]
        if xyz.size==0: return xyz
        if len(self.clip_pcts)==2:
            lo,hi = np.percentile(xyz[:,2], self.clip_pcts)
            xyz[:,2] = np.clip(xyz[:,2], lo, hi)
        return voxel_downsample(xyz, self.voxel)

    def rasterize(self) -> np.ndarray:
        xyz = self.get_concat_latest_xyz()
        self._pts_last = int(xyz.shape[0])
        bounds = (-self.L, self.L, -self.L, self.L)

    # keep a handle to the previous grid BEFORE overwriting it
        prev = None if (self.zmin_fixed is not None) else self._grid

    # bin points -> per-cell Z (in *meters*, not normalized)
        H = rasterize_agg(
            xyz, self.nrow, self.ncol, bounds,
            agg=self.aggregate,
            prev_grid=prev,
            smooth_passes=self.smooth_passes
        )

    # ---- NEW: enforce a ground plane for empty/invalid cells ----
        if self.zmin_fixed is not None:
            ground = float(self.zmin_fixed)
        # NaNs -> ground, and clamp any stray values below ground up to ground
            nanmask = ~np.isfinite(H)
            if nanmask.any():
                H[nanmask] = ground
            np.maximum(H, ground, out=H)
        else:
        # no fixed ground: fall back to previous grid, then to current min
            if prev is not None:
                mask = ~np.isfinite(H)
                if mask.any():
                    H[mask] = prev[mask]
            if np.isnan(H).any():
                H[np.isnan(H)] = np.nanmin(H)

    # optional flips for visual alignment
        if self.flip_y: H = np.flipud(H)
        if self.flip_x: H = np.fliplr(H)
        # Force the outer border to a fixed physical height (meters)
        EDGE_FLOOR = 0.20
        H[0,  :] = EDGE_FLOOR
        H[-1, :] = EDGE_FLOOR
        H[:,  0] = EDGE_FLOOR
        H[:, -1] = EDGE_FLOOR
        self._grid = H.astype(np.float32, copy=False)
        return self._grid


    def _stats(self):
        self.get_logger().info(f'pts={self._pts_last}  grid={self.nrow}x{self.ncol}  seq={self._seq}')

# ---------- TCP send loop ----------

def send_loop(node: MultiCloudNode, bind_host='0.0.0.0', bind_port=5005):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bind_host, bind_port))
    srv.listen(1)
    node.get_logger().info(f'TCP server listening on {bind_host}:{bind_port} (waiting for laptop viewer)')

    conn, addr = srv.accept()
    node.get_logger().info(f'Laptop connected from {addr}')

    interval = 1.0 / max(float(node.get_parameter("send_hz").value), 1.0)
    try:
        while rclpy.ok():
            H = node.rasterize()

            # Send RAW meters (no normalization)
            zmin = float(np.nanmin(H)); zmax = float(np.nanmax(H))
            if not np.isfinite(zmin) or not np.isfinite(zmax):
                time.sleep(interval); continue

            payload = zlib.compress(H.astype(np.float32, copy=False).tobytes())

            # Keep zmin/zmax in header for FYI; viewer will choose its own scaling
            header = struct.pack(
                '<4sBHHIffI', MAGIC, VERSION, node.nrow, node.ncol,
                node._seq, float(zmin), float(zmax), len(payload)
            )
            conn.sendall(header + payload)
            node._seq += 1
            time.sleep(interval)
    except (BrokenPipeError, ConnectionResetError):
        node.get_logger().warn('Laptop disconnected')
    finally:
        conn.close(); srv.close()

def main():
    rclpy.init()
    node = MultiCloudNode()
    t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t.start()
    try:
        send_loop(node)
    finally:
        node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()