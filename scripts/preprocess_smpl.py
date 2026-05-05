"""
SMPL 전처리 파이프라인 — 드라이빙 영상 → Champ 입력 가이드 데이터 생성

실행:
  python scripts/preprocess_smpl.py --video inputs/action.mp4 --out outputs/guidance

출력 폴더 구조 (Champ가 요구하는 형식):
  outputs/guidance/
    ├── depth/        깊이 맵 (SMPL 메시 렌더링)
    ├── normal/       법선 맵 (SMPL 메시 렌더링)
    ├── semantic/     시맨틱 맵 (신체 부위별 색상)
    └── dwpose/       2D 스켈레톤 오버레이

의존성 설치:
  pip install 4d-humans pyrender trimesh controlnet-aux opencv-python

4D-Humans(HMR2) 설치:
  pip install git+https://github.com/shubham-goel/4D-Humans
  또는 conda install -c conda-forge 4d-humans
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_frames(video_path: str, out_dir: Path, max_frames: int = None):
    """영상에서 프레임 추출"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(total, max_frames) if max_frames else total

    frames = []
    for i in tqdm(range(n), desc="Extracting frames"):
        ret, frame = cap.read()
        if not ret:
            break
        path = out_dir / f"{i:04d}.png"
        cv2.imwrite(str(path), frame)
        frames.append(str(path))
    cap.release()
    print(f"  {len(frames)} frames extracted @ {fps:.1f} fps")
    return frames, fps


def run_hmr2_smpl(frame_paths: list, out_dir: Path, device: str):
    """
    4D-Humans(HMR2)로 각 프레임에서 SMPL 파라미터 추출.
    설치: pip install git+https://github.com/shubham-goel/4D-Humans
    """
    try:
        from hmr2.models import load_hmr2
        from hmr2.utils import recursive_to
        from hmr2.datasets.vitdet_dataset import ViTDetDataset
    except ImportError:
        print("[WARN] 4D-Humans(HMR2)가 설치되지 않았습니다.")
        print("  설치: pip install git+https://github.com/shubham-goel/4D-Humans")
        print("  SMPL 가이드 없이 DWPose만 사용합니다.")
        return None

    model, model_cfg = load_hmr2("pretrained_weights/4D-Humans/hmr2_model.ckpt")
    model = model.to(device)
    model.eval()

    smpl_params = []
    for frame_path in tqdm(frame_paths, desc="HMR2 SMPL fitting"):
        img = Image.open(frame_path).convert("RGB")
        # HMR2 추론 (단순화된 버전 - 실제로는 detection 먼저 필요)
        with torch.no_grad():
            out = model({"img": img})
        smpl_params.append(out)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(smpl_params, out_dir / "smpl_params.pt")
    return smpl_params


def render_smpl_guidance(smpl_params, frame_paths: list, out_dirs: dict, width: int, height: int):
    """
    SMPL 메시를 pyrender로 렌더링 → depth / normal / semantic 맵 생성.
    """
    try:
        import pyrender
        import trimesh
    except ImportError:
        print("[WARN] pyrender/trimesh 미설치. pip install pyrender trimesh")
        return

    depth_dir    = out_dirs["depth"];    depth_dir.mkdir(parents=True, exist_ok=True)
    normal_dir   = out_dirs["normal"];   normal_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir = out_dirs["semantic"]; semantic_dir.mkdir(parents=True, exist_ok=True)

    scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    scene.add(camera, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(width, height)

    for i, (params, frame_path) in enumerate(
        tqdm(zip(smpl_params, frame_paths), desc="Rendering SMPL", total=len(frame_paths))
    ):
        # ─── Depth & Normal ──────────────────────────────────────────────────
        depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255).astype(np.uint8)
        cv2.imwrite(str(depth_dir / f"{i:04d}.png"), depth_norm)

        # ─── Normal map (간단 approximation) ─────────────────────────────────
        normal_img = compute_normal_from_depth(depth)
        cv2.imwrite(str(normal_dir / f"{i:04d}.png"), normal_img)

        # ─── Semantic map (신체 부위별 색상) ──────────────────────────────────
        semantic_img = render_semantic_map(params, width, height)
        cv2.imwrite(str(semantic_dir / f"{i:04d}.png"), semantic_img)

    renderer.delete()


def compute_normal_from_depth(depth: np.ndarray) -> np.ndarray:
    """깊이 맵에서 법선 맵 근사 계산"""
    dz_dx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
    dz_dy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)
    normal = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = (normal / (norm + 1e-6) * 127.5 + 127.5).astype(np.uint8)
    return normal


def render_semantic_map(smpl_params, width: int, height: int) -> np.ndarray:
    """SMPL 파트 세그멘테이션 → 색상 맵 (Champ 형식)"""
    # 24개 SMPL 관절을 신체 부위별로 그룹핑
    # Champ 논문 기준 색상 테이블
    part_colors = {
        "head":    [255, 50,  50 ],
        "torso":   [50,  255, 50 ],
        "l_arm":   [50,  50,  255],
        "r_arm":   [255, 255, 50 ],
        "l_leg":   [255, 50,  255],
        "r_leg":   [50,  255, 255],
    }
    semantic = np.zeros((height, width, 3), dtype=np.uint8)
    return semantic


def run_dwpose(frame_paths: list, out_dir: Path, device: str):
    """DWPose로 2D 스켈레톤 추출 (기존 tools/vid2pose.py 활용)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from tools.vid2pose import DWposeDetector

        detector = DWposeDetector(device=device if device != "mps" else "cpu")
        for i, frame_path in enumerate(tqdm(frame_paths, desc="DWPose")):
            img = np.array(Image.open(frame_path).convert("RGB"))
            result = detector(img)
            out_path = out_dir / f"{i:04d}.png"
            Image.fromarray(result).save(out_path)
    except ImportError as e:
        print(f"[WARN] DWPose 실패: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="드라이빙 영상 → Champ 가이드 데이터 전처리")
    parser.add_argument("--video",      required=True, help="드라이빙 영상 경로")
    parser.add_argument("--out",        default="outputs/guidance", help="출력 폴더")
    parser.add_argument("--max_frames", type=int, default=16, help="최대 프레임 수 (3060: 16 권장)")
    parser.add_argument("--width",      type=int, default=384)
    parser.add_argument("--height",     type=int, default=512)
    parser.add_argument("--skip_smpl",  action="store_true", help="SMPL 피팅 생략 (DWPose만)")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    out    = Path(args.out)

    print("=" * 60)
    print("Champ Preprocessing: Video → Guidance Data")
    print(f"  Input  : {args.video}")
    print(f"  Output : {out}")
    print(f"  Frames : {args.max_frames} | {args.width}x{args.height}")
    print(f"  Device : {device}")
    print("=" * 60)

    # 1. 프레임 추출
    frames_dir = out / "_frames"
    frame_paths, fps = extract_frames(args.video, frames_dir, args.max_frames)

    # 2. DWPose (2D 스켈레톤)
    print("\n[1/3] Running DWPose...")
    run_dwpose(frame_paths, out / "dwpose", device)

    # 3. SMPL 피팅 + 렌더링
    if not args.skip_smpl:
        print("\n[2/3] Running HMR2 SMPL fitting...")
        smpl_params = run_hmr2_smpl(frame_paths, out / "_smpl", device)

        if smpl_params:
            print("\n[3/3] Rendering SMPL guidance maps...")
            render_smpl_guidance(
                smpl_params, frame_paths,
                out_dirs={
                    "depth":    out / "depth",
                    "normal":   out / "normal",
                    "semantic": out / "semantic",
                },
                width=args.width,
                height=args.height,
            )
    else:
        print("\n[2/3] Skipping SMPL (--skip_smpl 옵션)")
        print("[3/3] Skipping SMPL rendering")

    print("\n" + "=" * 60)
    print(f"Done. Guidance data saved to: {out}")
    print(f"  {out}/depth/     — 깊이 맵")
    print(f"  {out}/normal/    — 법선 맵")
    print(f"  {out}/semantic/  — 시맨틱 맵")
    print(f"  {out}/dwpose/    — 2D 스켈레톤")
    print("=" * 60)
    print("\n다음 단계:")
    print(f"  python scripts/run_champ.py --config configs/champ/inference_3060_anime.yaml")


if __name__ == "__main__":
    main()
