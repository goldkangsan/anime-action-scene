"""
Anime Action Scene Generator — Gradio Web UI
실행: python run_gradio.py
      → http://localhost:7860 에서 열기

베이스 모델:
  - SD 1.5 (original): 일반 캐릭터 애니메이션
  - Anything V5 (anime): 애니 특화 스타일 (--anime 가중치 다운 필요)

M5 MacBook 포함 Apple Silicon(MPS) 자동 지원.
"""

import subprocess
import sys
import tempfile
import torch
import gradio as gr
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from tools.vid2pose import extract_pose_from_video
from scripts.pose2vid import load_models, build_pipeline, run_inference, get_device, get_weight_dtype


MODEL_CONFIGS = {
    "SD 1.5 (Original)":    "./configs/prompts/animation.yaml",
    "Anything V5 (Anime)":  "./configs/prompts/animation_anime.yaml",
}


def apply_speed(video_path: str, speed: float) -> str:
    """ffmpeg로 영상 배속 처리. speed=1.0이면 원본 그대로 반환."""
    if abs(speed - 1.0) < 0.01:
        return video_path
    out_path = str(Path(video_path).with_suffix("")) + f"_x{speed:.1f}.mp4"
    # setpts: 영상 빠르게, atempo: 오디오 빠르게 (오디오 없어도 오류 안남)
    pts = f"setpts={1.0/speed:.4f}*PTS"
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", pts, "-an", out_path],
        check=True,
        capture_output=True,
    )
    return out_path


class AnimateController:
    def __init__(self):
        self.pipe = None
        self.loaded_model = None
        self.device = get_device()
        self.weight_dtype = get_weight_dtype(self.device)
        print(f"[Device] {self.device} | dtype: {self.weight_dtype}")

    def load(self, model_name: str):
        """모델이 바뀌었을 때만 재로드."""
        if self.pipe is not None and self.loaded_model == model_name:
            return
        config_path = MODEL_CONFIGS[model_name]
        print(f"Loading models from: {config_path}")
        config = OmegaConf.load(config_path)
        vae, ref_unet, denoise_unet, pose_guider, image_enc, infer_config = load_models(
            config, self.weight_dtype, self.device
        )
        self.pipe = build_pipeline(
            vae, image_enc, ref_unet, denoise_unet, pose_guider,
            infer_config, self.weight_dtype, self.device
        )
        self.loaded_model = model_name
        self.config = config
        print("Models ready.")

    def animate(
        self,
        ref_image_path: str,
        driving_video_path: str,
        model_name: str,
        width: int,
        height: int,
        n_frames: int,
        steps: int,
        cfg: float,
        seed: int,
        speed: float,
    ):
        if ref_image_path is None or driving_video_path is None:
            return None, "입력 파일을 모두 업로드해주세요."

        self.load(model_name)

        # ─── 포즈 추출 ────────────────────────────────────────────────────────
        pose_video_path = str(
            Path(driving_video_path).parent
            / (Path(driving_video_path).stem + "_kps.mp4")
        )
        print(f"Extracting pose: {driving_video_path}")
        extract_pose_from_video(
            video_path=driving_video_path,
            output_path=pose_video_path,
            device=self.device if self.device != "mps" else "cpu",  # DWPose ONNX는 CPU/CUDA만 지원
        )

        # ─── inference ────────────────────────────────────────────────────────
        save_dir = Path("output/gradio") / datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir.mkdir(parents=True, exist_ok=True)

        generator = torch.Generator()
        generator.manual_seed(seed)

        out_path = run_inference(
            pipe=self.pipe,
            ref_image_path=ref_image_path,
            pose_video_path=pose_video_path,
            width=width,
            height=height,
            n_frames=n_frames,
            steps=steps,
            cfg=cfg,
            generator=generator,
            save_dir=save_dir,
        )

        # ─── 배속 후처리 ──────────────────────────────────────────────────────
        try:
            out_path = apply_speed(out_path, speed)
        except Exception as e:
            print(f"[WARN] 배속 처리 실패 (ffmpeg 필요): {e}")

        return out_path, f"완료: {out_path}"


controller = AnimateController()


# ─── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="Anime Action Scene Generator") as demo:
    gr.Markdown("## Anime Action Scene Generator")
    gr.Markdown(
        "캐릭터 이미지 1장 + 액션 영상을 업로드하면 애니 스타일 액션씬 영상을 생성합니다.  \n"
        f"**실행 환경**: `{controller.device}` | dtype: `{controller.weight_dtype}`"
    )

    with gr.Row():
        with gr.Column():
            ref_image    = gr.Image(type="filepath", label="캐릭터 이미지 (Reference)")
            drive_video  = gr.Video(label="액션 영상 (Driving Video)")

            model_choice = gr.Radio(
                choices=list(MODEL_CONFIGS.keys()),
                value="Anything V5 (Anime)",
                label="베이스 모델",
            )

            with gr.Accordion("생성 설정", open=True):
                speed    = gr.Slider(0.5, 3.0, value=1.5, step=0.25, label="배속 (1.0 = 원본 속도)")
                cfg      = gr.Slider(1.0, 10.0, value=3.5, step=0.5, label="CFG Scale (높을수록 포즈에 충실)")
                n_frames = gr.Slider(16, 64, value=32, step=8, label="생성 프레임 수")

            with gr.Accordion("고급 설정", open=False):
                width    = gr.Slider(384, 768, value=512,  step=64,  label="Width")
                height   = gr.Slider(512, 1024, value=784, step=64,  label="Height")
                steps    = gr.Slider(10, 50,  value=30,  step=5,   label="DDIM Steps")
                seed     = gr.Number(value=42, label="Seed", precision=0)

            run_btn = gr.Button("Generate Action Scene", variant="primary", size="lg")

        with gr.Column():
            output_video = gr.Video(label="결과 (ref | pose | generated)")
            status_text  = gr.Textbox(label="상태", interactive=False)

    run_btn.click(
        fn=controller.animate,
        inputs=[ref_image, drive_video, model_choice,
                width, height, n_frames, steps, cfg, seed, speed],
        outputs=[output_video, status_text],
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
