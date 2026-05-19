# Anime Action Scene Generator
**KUBIG Conference — Multimodal Team**

캐릭터 이미지 1장 + 액션 영상 → 애니 스타일 액션씬 영상 생성

---

## 실험 파이프라인

| | 2D Baseline | 3D Champ |
|---|---|---|
| **Guidance** | DWPose 2D skeleton | DWPose + Depth + Normal |
| **Base model** | SD 1.5 | SD 1.5 |
| **Anime model** | Anything V5 | Anything V5 |
| **Colab notebook** | `colab_2d_baseline.ipynb` | `colab_champ_inference.ipynb` |

---

## Colab 실행

### 2D Baseline (Moore-AnimateAnyone)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/goldkangsan/anime-action-scene/blob/main/colab_2d_baseline.ipynb)

### 3D Champ
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/goldkangsan/anime-action-scene/blob/main/colab_champ_inference.ipynb)

---

## 실행 순서 (공통)

```
셀 0 → 셀 1 → 셀 2 (pip 설치, 자동 재시작)
→ 셀 0 → 셀 1 → 셀 3 (import 테스트) → 이후 순서대로
```
> ⚠️ 셀 2 이후 런타임이 자동 재시작됨 — 정상 동작. 셀 2만 건너뛰고 다시 실행.

---

## 디렉토리 구조

```
anime-action-scene/
├── src/                        # Moore-AnimateAnyone 핵심 모듈
│   ├── dwpose/                 # DWPose 2D skeleton detector
│   ├── models/                 # UNet2D/3D, PoseGuider
│   ├── pipelines/              # Pose2VideoPipeline
│   └── utils/
├── champ/                      # Champ submodule (3D pipeline)
├── scripts/
│   ├── pose2vid.py             # 2D inference
│   ├── preprocess_smpl.py      # MiDaS depth + DWPose + normal 추출
│   └── run_champ.py            # 3D Champ inference wrapper
├── tools/
│   ├── download_weights.py     # 가중치 다운로드 (--anime, --champ 옵션)
│   └── vid2pose.py             # 영상 → DWPose 영상 변환
├── configs/
│   ├── prompts/                # 2D inference configs
│   │   ├── animation.yaml      # SD 1.5
│   │   └── animation_anime.yaml # Anything V5
│   └── champ/                  # 3D Champ configs
│       ├── colab_t4.yaml       # T4/A100, SD 1.5
│       └── colab_t4_anime.yaml # T4/A100, Anything V5
├── colab_2d_baseline.ipynb     # ← 2D 실험용
└── colab_champ_inference.ipynb # ← 3D 실험용
```

---

## 검증된 패키지 버전

```
diffusers==0.24.0
transformers>=4.36.0,<4.40.0
huggingface_hub>=0.19.0,<0.23.0   ← 0.23+ 에서 cached_download 삭제됨
accelerate
```
