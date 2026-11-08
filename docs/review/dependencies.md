# Dependency inventory

Everything ASPIRE needs, by layer. Versions are taken from `pyproject.toml`, `uv.lock`,
`configs/dependency-lock.json` and the bootstrap scripts at release `fa00f0e`.

## 1. Host

| Item | Requirement | Notes |
| --- | --- | --- |
| OS / arch | Linux x86_64 | `uv.lock` resolves only `sys_platform == 'linux' and platform_machine == 'x86_64'` |
| Python | 3.10–3.12 (`requires-python >=3.10,<3.13`); **3.11 exactly** for BEHAVIOR | Some distros default `python3` to 3.13. Pass `--python python3.11` / `CAP_HARNESS_PYTHON`. |
| uv | Any recent version | Every bootstrap uses `uv sync --frozen` |
| git, git-lfs | Submodules, LFS assets | |
| ffmpeg | Run videos, through `imageio[ffmpeg]` wheels; system copy optional | |
| OpenGL (headless) | NVIDIA driver + EGL (GPU), **or** Mesa `libosmesa6` (CPU) | `MUJOCO_GL=egl` fails without an NVIDIA device. Use `osmesa` on CPU. |
| NVIDIA GPU | Ada sm_89 (RTX 4090, L40) or Blackwell sm_120 (RTX 5090; RTX PRO 6000 via profile `rtxpro6000`, added on this branch) | Ampere sm_86 needs a new arch entry (NVlabs/ASPIRE#28). |
| CUDA | Runtime 12.8 via torch cu128 wheels. A system `nvcc` 12.x is needed **only** for BEHAVIOR's in-process cuRobo. | The cuRobo service bootstrap pulls its own `nvidia-cuda-nvcc` wheel. |
| Disk | About 20 GB for the LIBERO + Robosuite + providers venvs; about 52 GB extra for BEHAVIOR (12 GB venv + 40 GB data) | `.venv-libero` alone is about 7 GB (torch + CUDA wheels). |
| RAM | 16 GB is fine for MuJoCo benchmarks. BEHAVIOR docs say 16–32 GB. | |
| VRAM | Isaac Sim about 12 GB, plus SAM3 and Contact-GraspNet about 8 GB | A 24 GB card is tight for BEHAVIOR plus providers. |

## 2. Python environments

Each component has its own venv (`configs/environments.json`). Simulators install client-only
dependencies and reach models over HTTP.

| Venv | Built by | Key pins / packages |
| --- | --- | --- |
| core (all venvs) | — | `numpy==1.26.4`, `scipy>=1.12,<1.18`, `PyYAML>=6`, `requests>=2.32` |
| `.venv-libero` | `scripts/bootstrap_libero.sh` | `libero` (third_party/LIBERO-PRO, editable), `robosuite` (third_party/robosuite fork), `mujoco==3.3.0`, `robomimic==0.2.0`, `bddl==1.0.1`, `gym==0.25.2`, `gymnasium>=0.29`, `torch==2.9.1+cu128`, `torchvision==0.24.1`, `opencv-python-headless<4.13`, `easydict==1.9`, `einops==0.4.1`, `thop`, `future==0.18.2`, `cloudpickle==2.1.0`, `PyOpenGL==3.1.6` |
| `.venv-robosuite` | `scripts/bootstrap_robosuite.sh` | `robosuite` (fork, editable), `mujoco==3.3.0`, `gym==0.25.2`, `numba`, `opencv-python-headless`, `imageio[ffmpeg]` (no torch, about 0.3 GB) |
| `.venv-sam3` | `scripts/bootstrap_sam3.sh` | `sam3` (third_party/sam3 fork), `torch==2.9.1`, `torchvision==0.24.1`, `open_clip_torch`, `decord`, `pycocotools`, `fastapi`, `uvicorn`, `transformers<5` |
| `.venv-graspnet` | `scripts/bootstrap_graspnet.sh` | `contact-graspnet-pytorch` (submodule; builds `pointnet2` CUDA ops against torch), `open3d>=0.19`, `trimesh`, `pyrender` |
| `.venv-pyroki` | `scripts/bootstrap_pyroki.sh` | `pyroki` @ `388e43e`, `jax<0.4.30`, `jaxls`, `jaxlie`, `robot_descriptions` (downloads `panda_description` on first start), `fastapi`, `uvicorn` |
| `.venv-curobo` | `scripts/bootstrap_curobo.sh` | cuRobo @ `a35a708`, compiled from source with CUDA extensions for the profile's arch |
| `.venv-behavior` | `scripts/bootstrap_behavior.sh` | Python 3.11, `isaacsim[all,extscache]==5.1.0` (pypi.nvidia.com), OmniGibson / BEHAVIOR-1K v3.9.2 @ `b197991`, `torch==2.7.0+cu128`, `numpy<2`, in-process cuRobo |
| `agent` extra | `scripts/bootstrap_system2.sh` | `mcp>=1.26,<2`, `fastapi`, `uvicorn`, `viser[urdf]>=1.0.30,<2`, `pillow`, `jsonschema` |
| `yam-real` extra | `scripts/bootstrap_yam_real.sh` | `portal>=3.7.3` (RPC), `mink>=0.0.4` (IK), `mujoco==3.3.0`, `pyrealsense2>=2.55`, `imageio[ffmpeg]` |
| `yam-real-server` extra | | `damiao-motor>=1.0.7b1` (python-can / SocketCAN) |
| `dev` extra | | `pytest>=8`, `pytest-timeout`, `ruff>=0.12` |
| `.venv-dev` (this branch) | `scripts/bootstrap_cpu_dev.sh` | dev + agent + yam-real + robosuite. Collects and runs the whole test suite. |

## 3. Submodules (`third_party/`, pinned in `configs/dependency-lock.json`)

| Path | Upstream | Commit | Needed for |
| --- | --- | --- | --- |
| LIBERO-PRO | github.com/uynitsuj/LIBERO-PRO (fork) | `47aaa80` | LIBERO-Pro |
| robosuite | github.com/Max-Fu/robosuite (fork of 1.4) | `a498b08` | LIBERO **and** Robosuite |
| sam3 | github.com/Max-Fu/sam3 (fork) | `6fe87d6` | SAM3 provider |
| contact_graspnet_pytorch | github.com/uynitsuj/contact_graspnet_pytorch | `da3dcfb` | Grasp provider |
| curobo | github.com/NVlabs/curobo | `a35a708` | cuRobo provider (optional) |
| BEHAVIOR-1K | github.com/StanfordVL/BEHAVIOR-1K v3.9.2 | `b197991` | BEHAVIOR only (large) |

## 4. Services and ports

| Service | Port | Started by |
| --- | --- | --- |
| SAM3 | 8114 | `scripts/supervise_services.sh --providers sam3` |
| Contact-GraspNet | 8115 | `--providers contact_graspnet` |
| PyRoki | 8116 | `--providers pyroki`, or directly with `JAX_PLATFORMS=cpu python -m cap_harness.providers.pyroki.service` |
| cuRobo | 8118 | `--providers curobo` |
| System-2 MCP server | 8222 (`/mcp`, `/health`, `/steer`) | `scripts/start_system2.sh [--sim] [--allow-motion]` |
| YAM arm servers (portal RPC) | per `station.yaml` | `yam-servers` (binds 0.0.0.0, no authentication — firewall them) |

The supervisor runs in the foreground and kills every provider when its shell exits. Use
`setsid nohup … & disown` for long runs (`docs/environments.md`).

## 5. Credentials and accounts

| What | Why |
| --- | --- |
| Hugging Face token with access to `facebook/sam3` | The SAM3 weights are gated, and the download fails (401) at first service load, not at bootstrap. |
| Anthropic or OpenAI account (Claude Code / Codex) | Drives the learning loop. Budget roughly $10–$60 per task (paper Tables 18–23). |
| BEHAVIOR Data Bundle ToS (`--accept-dataset-tos`), Omniverse EULA | BEHAVIOR only |

## 6. Licences to check before use

| Component | Licence | Implication |
| --- | --- | --- |
| ASPIRE / cap_harness | Apache-2.0 | |
| Contact-GraspNet | NVIDIA Source Code License: **non-commercial research and evaluation only** | Replace it for any commercial use. |
| BEHAVIOR Data Bundle | **Non-commercial academic research**; no redistribution | |
| SAM3 | SAM License (Meta) | Check the restrictions; acknowledgement is required in publications. |
| cuRobo | Apache-2.0, per `dependency-lock.json` | Older cuRobo releases used a non-commercial licence; confirm for the pinned commit. |
| LIBERO-PRO, robosuite, PyRoki, Mink, MuJoCo | MIT / Apache-2.0 | |
| Isaac Sim | NVIDIA Omniverse EULA | |

## 7. Real-robot hardware (YAM reference station)

- 2× I2RT YAM arms with DaMiao motors on SocketCAN (`can_follow_l`, `can_follow_r`). No udev or
  bitrate setup ships with the repo.
- Intel RealSense (D405/D435 class) cameras, with RealSense udev rules installed.
- A ChArUco board for hand-eye calibration
  (`python -m cap_harness.yam_real.calibration`).
- Codex CLI for the System-2 agent (`scripts/start_system2_agent.sh`).
