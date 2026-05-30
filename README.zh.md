# AI Cover + Pop2Piano 安装与运行指南

英文简要说明见 `README.md`。旧的 audio-to-MIDI / PianistTransformer 流程见 `docs/audio_to_midi_legacy.zh.md`。

这个仓库当前主线是 cover 工作流：输入一首歌，生成 DDSP-SVC 转换后的人声、Pop2Piano 钢琴伴奏，以及最终混音 WAV。

主入口：

```bash
python scripts/run_ai_piano_cover.py ...
```

## 工作流概览

`scripts/run_ai_piano_cover.py` 会依次执行：

1. 用 `ffmpeg` 把输入音频统一成 `44100 Hz` 双声道 WAV。
2. 用 Demucs 分离 `vocals.wav` 和 `no_vocals.wav`。
3. 用 DDSP-SVC 把人声转换成目标声线。
4. 用 Pop2Piano 从原曲生成钢琴伴奏 MIDI。
5. 用 FluidSynth + SoundFont 把钢琴 MIDI 渲染成 WAV。
6. 用 `ffmpeg` 混合转换后人声和钢琴伴奏。

默认输出目录是 `assets/output/cover/`：

- `preprocessed/<stem>.wav`
- `separated/<stem>_vocals.wav`
- `separated/<stem>_no_vocals.wav`
- `vocals/<stem>_ddsp_vocals.wav`
- `piano/<stem>_pop2piano.mid`
- `piano/<stem>_pop2piano.wav`
- `final/<stem>_ai_cover_piano.wav`

## 1. 克隆仓库

```bash
git clone <repo-url>
cd DeepLearning-Project
```

这个仓库包含 vendor 代码目录，但大模型、音色和部分 runtime 文件不要提交到 GitHub。它们应通过外部拷贝、共享盘、release asset 或系统包安装准备。

## 2. 创建 Conda 环境

cover 工作流只推荐使用 `environment.pianoformer-cover.yml`：

```bash
conda env create -f environment.pianoformer-cover.yml
conda activate pianoformer-cover
```

确认环境可用：

```bash
python - <<'PY'
import torch, torchaudio, librosa, soundfile, yaml, transformers, pretty_midi, midi2audio
print("imports_ok")
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("torch_cuda:", torch.version.cuda)
PY
```

目标机器推荐是 Linux + NVIDIA GPU。普通 shell 中 `torch.cuda.is_available()` 如果是 `False`，先确认当前 shell 是否真的拿到了 GPU；在集群环境里通常需要进入 GPU allocation 后再检查。

## 3. 准备外部资源

下面资源需要从已有机器外部拷贝到新机器，或者按各小节用系统包/联网方式准备。建议保持 README 中的相对路径，这样命令可以直接复用。

### 3.1 DDSP-SVC 和 Demucs

需要存在：

```text
third_party/ai_cover/demucs/
third_party/ai_cover/DDSP-SVC/
third_party/ai_cover/DDSP-SVC/exp/reflow-test/config.yaml
third_party/ai_cover/DDSP-SVC/exp/reflow-test/model_30000.pt
third_party/ai_cover/DDSP-SVC/pretrain/contentvec/checkpoint_best_legacy_500.pt
third_party/ai_cover/DDSP-SVC/pretrain/rmvpe/model.pt
third_party/ai_cover/DDSP-SVC/pretrain/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/config.json
third_party/ai_cover/DDSP-SVC/pretrain/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt
```

当前代码默认使用：

```text
third_party/ai_cover/DDSP-SVC/exp/reflow-test/model_30000.pt
```

如果模型放在别处，运行时传：

```bash
--ddsp-model-ckpt /path/to/model_30000.pt
```

### 3.2 Pop2Piano 模型

推荐把 Hugging Face 模型 `sweetcocoa/pop2piano` 外部拷贝到仓库内的本地目录，例如：

```text
models/pop2piano/sweetcocoa-pop2piano/
```

运行时显式传模型目录：

```bash
--pop2piano-model models/pop2piano/sweetcocoa-pop2piano
```

如果新机器能稳定访问 Hugging Face，也可以直接传：

```bash
--pop2piano-model sweetcocoa/pop2piano
```

### 3.3 FluidSynth 和 SoundFont

渲染钢琴 WAV 需要两类资源：

1. `fluidsynth` 可执行文件
2. `.sf2` SoundFont 音色文件

不要把 SoundFont、`.deb` 包、解压后的 runtime libs 提交到 GitHub；这些文件体积较大，也不是源码。推荐二选一准备。

方案 A：从已有机器外部拷贝本地免 sudo 资源包，保持这个布局：

```text
assets/soundfonts/extracted/usr/share/sounds/sf2/FluidR3_GM.sf2
assets/soundfonts/fluidsynth_pkg/usr/bin/fluidsynth
assets/soundfonts/runtime_libs/usr/lib/x86_64-linux-gnu/
```

其中真正传给 `--soundfont` 的必须是 `.sf2` 文件：

```text
assets/soundfonts/extracted/usr/share/sounds/sf2/FluidR3_GM.sf2
```

不要误传成目录：

```text
assets/soundfonts/extracted/usr/share/sounds
```

方案 B：如果有 sudo 权限，也可以直接系统安装：

```bash
sudo apt-get update
sudo apt-get install -y fluidsynth fluid-soundfont-gm
```

常见系统路径是：

```text
/usr/bin/fluidsynth
/usr/share/sounds/sf2/FluidR3_GM.sf2
```

本仓库命令默认使用方案 A。先在仓库根目录导出路径，后面的命令都复用这三个变量：

```bash
export SOUNDFONT="$PWD/assets/soundfonts/extracted/usr/share/sounds/sf2/FluidR3_GM.sf2"
export FLUIDSYNTH_BIN="$PWD/assets/soundfonts/fluidsynth_pkg/usr/bin/fluidsynth"
export FLUIDSYNTH_LIB_DIR="$PWD/assets/soundfonts/runtime_libs/usr/lib/x86_64-linux-gnu"
```

不要手动把 `assets/soundfonts/extracted/usr/share/sounds/sf2/FluidR3_GM.sf2` 拆成两行输入；否则 shell 会把 `sf2/FluidR3_GM.sf2` 当成一个新命令执行。

检查 FluidSynth：

```bash
LD_LIBRARY_PATH="$FLUIDSYNTH_LIB_DIR" "$FLUIDSYNTH_BIN" --version
```

如果目标机器已经系统安装了 `fluidsynth`，也可以运行时只传系统 soundfont，或者省略 `--fluidsynth-bin`。

## 4. 资源完整性检查

在正式跑模型前，先确认关键文件存在：

```bash
test -f third_party/ai_cover/DDSP-SVC/exp/reflow-test/model_30000.pt
test -f third_party/ai_cover/DDSP-SVC/exp/reflow-test/config.yaml
test -f third_party/ai_cover/DDSP-SVC/pretrain/contentvec/checkpoint_best_legacy_500.pt
test -f third_party/ai_cover/DDSP-SVC/pretrain/rmvpe/model.pt
test -f third_party/ai_cover/DDSP-SVC/pretrain/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt
test -f "$SOUNDFONT"
test -x "$FLUIDSYNTH_BIN"
test -d "$FLUIDSYNTH_LIB_DIR"
```

再做一次 dry-run。dry-run 会打印每个阶段实际执行的命令，不会跑模型：

```bash
python scripts/run_ai_piano_cover.py \
  --input sample.mp3 \
  --device cuda \
  --pop2piano-device cuda \
  --pop2piano-model models/pop2piano/sweetcocoa-pop2piano \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR" \
  --dry-run
```

## 5. 正式运行

把输入音频放在仓库内或传绝对路径，然后运行：

```bash
python scripts/run_ai_piano_cover.py \
  --input yishiki.mp3 \
  --output-root assets/output/cover \
  --device cuda \
  --pop2piano-device cuda \
  --pop2piano-model models/pop2piano/sweetcocoa-pop2piano \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

成功后重点检查：

```text
assets/output/cover/final/sample_ai_cover_piano.wav
```

如果 GPU 显存不足，可以先把 Pop2Piano 放到 CPU：

```bash
--pop2piano-device cpu
```

如果整条链路都要 CPU smoke test：

```bash
python scripts/run_ai_piano_cover.py \
  --input sample.mp3 \
  --output-root assets/output/cover_cpu_test \
  --device cpu \
  --pop2piano-device cpu \
  --pop2piano-model models/pop2piano/sweetcocoa-pop2piano \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

CPU 可以用于验证安装，但速度会明显慢于 GPU。

## 常用参数

- `--input`: 输入音频路径，支持 `mp3/wav/flac` 等 `ffmpeg` 可读格式。
- `--output-root`: cover 输出根目录，默认 `assets/output/cover`。
- `--device`: Demucs 和 DDSP-SVC 使用的设备，`cuda` 或 `cpu`。
- `--spk-id`: DDSP-SVC speaker id，默认 `1`。
- `--key`: DDSP-SVC 半音移调，默认 `0`。
- `--pre-pitch-shift`: 预处理阶段先移调的半音数，例如 `4.5` 会在分离、转换和 Pop2Piano 前先升 4.5 个半音，默认 `0.0`。
- `--pitch-extractor`: DDSP-SVC pitch extractor，默认 `rmvpe`。
- `--ddsp-model-ckpt`: DDSP-SVC checkpoint 路径。
- `--pop2piano-model`: Pop2Piano Hugging Face id 或本地模型目录。
- `--pop2piano-composer`: Pop2Piano composer token，默认 `composer1`。
- `--pop2piano-device`: Pop2Piano 设备，`auto`、`cuda` 或 `cpu`。
- `--pop2piano-max-length`: Pop2Piano 生成 token 上限，默认 `256`。
- `--vocals-volume`: 最终混音人声音量，默认 `1.0`。
- `--piano-volume`: 最终混音钢琴音量，默认 `1.0`。

## 后端 API

CLI 入口仍然保留。前端联调用 FastAPI 后端：

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

后端启动时会读取 `configs/roles.yaml`，并按 exp 目录下已有的 8 个 DDSP checkpoint 做角色列表。默认 `preload_mode: torch_cpu` 会在启动时把 8 个 DDSP checkpoint 读入 CPU 内存，并校验 Demucs、Pop2Piano、SoundFont 和 FluidSynth 资源。如果只想做资源校验、不预读权重，可以设置：

```bash
export COVER_PRELOAD_MODE=validate
```

常用接口：

```text
GET  /api/health
GET  /api/config
GET  /api/models/status
POST /api/jobs
GET  /api/jobs/{job_id}
GET  /api/jobs/{job_id}/events
POST /api/jobs/{job_id}/cancel
POST /api/jobs/{job_id}/remix
GET  /api/jobs/{job_id}/files/input
GET  /api/jobs/{job_id}/files/vocals
GET  /api/jobs/{job_id}/files/piano
GET  /api/jobs/{job_id}/files/final
```

提交任务使用 `multipart/form-data`：

```text
audio: File
role_id: amoris
key: 0
vocals_volume: 1.0
piano_volume: 1.0
```

前端进度推送使用 SSE 连接：

```text
GET /api/jobs/{job_id}/events
```

只调整第四阶段音量时调用 remix，不会重跑分离、翻唱和 piano cover：

```http
POST /api/jobs/{job_id}/remix
Content-Type: application/json

{"vocals_volume": 0.9, "piano_volume": 0.6}
```

## 常见问题

### `torch.cuda.is_available()` 是 `False`

先确认目标机器有 NVIDIA driver，并且当前 shell 能看到 GPU：

```bash
nvidia-smi
```

如果在 Slurm 或类似集群上，需要先进入 GPU job，再激活 conda 环境并运行命令。

### Pop2Piano 尝试联网下载

如果 `--pop2piano-model` 指向本地存在的目录，脚本会设置离线环境变量并清理 proxy。确认路径写对：

```bash
test -d models/pop2piano/sweetcocoa-pop2piano
```

### DDSP-SVC 报缺少 checkpoint

确认 `config.yaml` 里引用的 encoder 和 vocoder 路径都存在。当前默认配置需要：

```text
pretrain/contentvec/checkpoint_best_legacy_500.pt
pretrain/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt
pretrain/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/config.json
```

### FluidSynth 或共享库缺失

export FLUIDSYNTH_BIN=assets/soundfonts/fluidsynth_pkg/usr/bin/fluidsynth

优先传完整三件套，并确保 `--soundfont` 是 `.sf2` 文件路径：

```bash
--soundfont "$SOUNDFONT" \
--fluidsynth-bin "$FLUIDSYNTH_BIN" \
--fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

也可以在系统层安装 FluidSynth 和 SoundFont 后，改用系统路径。

如果报：

```text
SoundFont path is not a file
```

说明传入的路径不是 `.sf2` 文件。检查：

```bash
find assets/soundfonts -type f -name '*.sf2' -print
```

当前推荐路径是：

```text
assets/soundfonts/extracted/usr/share/sounds/sf2/FluidR3_GM.sf2
```

### `ffmpeg` 报动态库错误

`environment.pianoformer-cover.yml` 已包含 conda 版 `ffmpeg`。确认正在使用 `pianoformer-cover` 环境：

```bash
which ffmpeg
ffmpeg -version
```

### Demucs 首次运行慢

Demucs 可能需要加载或缓存模型。GPU 机器上建议保留默认：

```bash
--device cuda
```

显存不足时再改成：

```bash
--device cpu
```
