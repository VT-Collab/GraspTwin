"""ZMQ REQ client for servers/camera_server.py (Orbbec RGB-D capture).

Usage:
    python -m semantic_grasp.camera --out-prefix frame

    from semantic_grasp.camera import OrbbecClient
    client = OrbbecClient()               # defaults to config.CAMERA
    bgr, depth, header = client.capture()

    client.record_start(task)        # RGB clip written on the CAMERA HOST
    ...                              # (camera_server.RECORD_DIR), `task` burned in
    status = client.record_stop()    # {"path", "frames", "duration_s", ...}
"""
import argparse
import json

import cv2
import numpy as np
import zmq

from .config import CAMERA


class OrbbecClient:
    def __init__(self, connect_addr=CAMERA, timeout_ms=10_000):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(connect_addr)

    def get_info(self):
        """Return the server's camera info dict (resolution, fps, intrinsics, etc.)."""
        self._sock.send(b"info")
        reply = json.loads(self._sock.recv())
        if not reply.get("ok"):
            raise RuntimeError(f"Server error: {reply.get('error')}")
        return reply

    def capture(self):
        """Return (bgr_image, depth_image, header_dict). Raises RuntimeError on server error."""
        self._sock.send(b"capture")
        parts = self._sock.recv_multipart()

        header = json.loads(parts[0])
        if not header.get("ok"):
            raise RuntimeError(f"Server error: {header.get('error')}")

        jpeg_bytes, depth_bytes = parts[1], parts[2]

        bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Failed to decode JPEG color image from server")

        depth = np.frombuffer(depth_bytes, dtype=np.dtype(header["depth_dtype"]))
        depth = depth.reshape(header["depth_shape"])

        # The pipeline treats depth as uint16 millimetres everywhere (perception
        # divides by 1000). That holds in the device's default precision mode,
        # but it can be put in a 0.1 mm mode, which would silently make every
        # distance 10x too large — loud beats silent.
        scale = header.get("depth_scale_mm", 1.0)
        if scale != 1.0:
            raise RuntimeError(
                f"camera reports depth_scale_mm={scale}, but this pipeline assumes "
                f"raw uint16 depth is already in millimetres. Either put the device "
                f"back in its 1 mm precision mode or teach perception to scale."
            )

        return bgr, depth, header

    def record_start(self, task, name=None):
        """Start an RGB recording ON THE SERVER'S disk (camera_server.RECORD_DIR
        on the lab PC) with `task` burned across the top of every frame;
        nothing streams back here. `name` is the clip's file stem (default: a
        timestamp + the task). Returns the recorder's status dict — "path" is
        where the clip lives on the camera host."""
        return self._json_cmd({"cmd": "record_start", "task": str(task), "name": name})

    def record_stop(self):
        """Stop the recording record_start began. Returns the status dict:
        "path", "frames", "duration_s", and "error" if the recorder hit one."""
        return self._json_cmd({"cmd": "record_stop"})

    def _json_cmd(self, req):
        self._sock.send(json.dumps(req).encode("utf-8"))
        reply = json.loads(self._sock.recv())
        if not reply.get("ok"):
            raise RuntimeError(f"Server error: {reply.get('error')}")
        return reply

    def close(self):
        self._sock.close()
        self._ctx.term()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    parser = argparse.ArgumentParser(description="ZMQ client for the Orbbec RGB-D server")
    parser.add_argument("--connect", default=CAMERA)
    parser.add_argument("--out-prefix", default="frame",
                        help="Save color/depth to <prefix>_color.png and <prefix>_depth.png")
    parser.add_argument("--info", action="store_true",
                        help="Just print camera info (resolution, intrinsics, etc.) and exit")
    args = parser.parse_args()

    with OrbbecClient(args.connect) as client:
        if args.info:
            print(json.dumps(client.get_info(), indent=2))
            return

        bgr, depth, header = client.capture()
        print("Received frame:", header)

        color_path = f"{args.out_prefix}_color.png"
        depth_path = f"{args.out_prefix}_depth.png"
        cv2.imwrite(color_path, bgr)
        cv2.imwrite(depth_path, depth)  # 16-bit PNG preserves raw depth values
        print(f"Saved {color_path} and {depth_path}")


if __name__ == "__main__":
    main()
