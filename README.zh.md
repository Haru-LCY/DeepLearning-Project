# 带可选 PianistTransformer 阶段的音频转 MIDI MVP

英文版文档：`README.md`

这个仓库现在提供一条面向 Linux 的单文件符号音乐流程，可以把输入音频转换成：

1. 转写得到的 MIDI
2. 清理后的 MIDI
3. 可选的 PianistTransformer 表情钢琴 MIDI
4. 最终钢琴 WAV

原始 baseline 流程仍然可以单独工作。现在默认推荐使用单一 `pianist-transformer` Python 环境来同时运行 baseline 和 PianistTransformer；如果你以后还想分环境，也仍然可以通过 `--pt-python` 显式指定另一个解释器。

## 当前状态

仓库现在支持基于 Python `3.11` 的单环境工作流，核心组件包括：

- `basic-pitch`
- `pretty_midi`
- `midi2audio`
- `FluidSynth`
- `PianistTransformer`
- PyTorch `2.7.1` CUDA `11.8` 官方 wheel

推荐的服务器使用方式是：

- 在 `pianist-transformer` 环境里直接跑完整 baseline pipeline
- 需要表情钢琴渲染时，加 `--enable-pianist-transformer`
- 默认使用当前 Python 解释器运行 PT 阶段

即使系统 CUDA 比 `11.8` 更新，只要驱动足够新，官方 `cu118` wheel 仍然是稳定可用的方案。

## 流程概览

baseline 路径：

1. 音频预处理成统一格式 WAV
2. 用 `basic-pitch` 转成原始 MIDI
3. 用 `pretty_midi` 清理过短音符
4. 用 `midi2audio` + `FluidSynth` 渲染成 WAV

可选表情渲染路径：

1. 音频预处理
2. 转写原始 MIDI
3. MIDI cleanup
4. 用 PianistTransformer 生成表情 raw MIDI
5. 把表情 timing 映射回 score 对齐的 MIDI
6. 用映射后的 MIDI 渲染 WAV

默认输出位置：

- 预处理 WAV：`assets/output/convert/<stem>.wav`
- 原始 MIDI：`assets/output/raw/<stem>_basic_pitch.mid`
- 清理后 MIDI：`assets/output/clean/<stem>_clean.mid`
- PT raw MIDI：`assets/output/expressive/raw/<stem>_pt_raw.mid`
- PT mapped MIDI：`assets/output/expressive/mapped/<stem>_pt_mapped.mid`
- 最终渲染 WAV：`assets/output/rendered/<stem>.wav`

主入口脚本：

- `python scripts/run_pipeline.py ...`
- `python scripts/run_expressive_render.py ...`

## 项目结构

```text
.
├── README.md
├── README.zh.md
├── environment.yml
├── requirements.txt
├── PianistTransformer/
├── configs/
├── assets/
├── scripts/
│   ├── run_cleanup_midi.py
│   ├── run_expressive_render.py
│   ├── run_pipeline.py
│   ├── run_preprocess_audio.py
│   ├── run_render.py
│   └── run_transcription.py
├── src/
│   ├── audio_preprocess.py
│   ├── expressive_render.py
│   ├── midi_cleanup.py
│   ├── render.py
│   ├── transcription.py
│   └── utils.py
└── tests/
```

关键目录说明：

- `assets/output/convert/`：统一格式单声道 `22050 Hz` WAV
- `assets/output/raw/`：`basic-pitch` 生成的原始 MIDI
- `assets/output/clean/`：清理后的 MIDI
- `assets/output/expressive/raw/`：PianistTransformer 生成的 raw 表情 MIDI
- `assets/output/expressive/mapped/`：默认用于渲染的 mapped/editable 表情 MIDI
- `assets/output/rendered/`：最终 WAV 输出

## 环境搭建

### 推荐的单环境方案

直接使用仓库提供的环境文件：

```bash
conda env create -f environment.yml
conda activate pianist-transformer
```

也可以手动创建：

```bash
conda create -n pianist-transformer python=3.11 pip -y
conda activate pianist-transformer
python -m pip install -r requirements.txt
```

说明：

- `requirements.txt` 现在同时包含 baseline 和 PianistTransformer 运行时依赖
- `numpy<2` 仍然保留，用来保障 `basic-pitch` 兼容性
- 文件里已经加入官方 PyTorch `cu118` 索引
- 在 Python `3.11` 下，`basic-pitch` 往往会安装 TensorFlow 路线，而不是旧的 `tflite-runtime` 轻量路线

建议验证 import：

```bash
python - <<'PY'
import basic_pitch
import pretty_midi
import midi2audio
import librosa
import soundfile
import torch
import transformers
import accelerate
import miditoolkit
import partitura
import numpy
print("python_deps_ok")
print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)
print(numpy.__version__)
PY
```

## FluidSynth 和 SoundFont 准备

渲染需要：

1. `fluidsynth` 可执行文件
2. `.sf2` soundfont

仓库已经支持本地免 sudo 运行时布局：

- soundfont：
  `assets/soundfonts/extracted/usr/share/sounds/sf2/FluidR3_GM.sf2`
- FluidSynth 二进制：
  `assets/soundfonts/fluidsynth_pkg/usr/bin/fluidsynth`
- 运行时库：
  `assets/soundfonts/runtime_libs/usr/lib/x86_64-linux-gnu`

建议导出：

```bash
export SOUNDFONT=assets/soundfonts/extracted/usr/share/sounds/sf2/FluidR3_GM.sf2
export FLUIDSYNTH_BIN=assets/soundfonts/fluidsynth_pkg/usr/bin/fluidsynth
export FLUIDSYNTH_LIB_DIR=$(pwd)/assets/soundfonts/runtime_libs/usr/lib/x86_64-linux-gnu
```

快速检查：

```bash
LD_LIBRARY_PATH="$FLUIDSYNTH_LIB_DIR" "$FLUIDSYNTH_BIN" --version
```

## Baseline 一条命令跑通

下面这条命令会在单一 `pianist-transformer` 环境里跑原始 4 阶段 pipeline：

```bash
python scripts/run_pipeline.py \
  --input sample.mp3 \
  --output-root assets/output \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

预期输出：

- `assets/output/convert/sample.wav`
- `assets/output/raw/sample_basic_pitch.mid`
- `assets/output/clean/sample_clean.mid`
- `assets/output/rendered/sample.wav`

## 在主 pipeline 中启用 PianistTransformer

加上这个开关即可启用表情渲染阶段：

```bash
python scripts/run_pipeline.py \
  --input sample.mp3 \
  --output-root assets/output \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR" \
  --enable-pianist-transformer
```

默认情况下，PT 阶段直接使用当前 Python 解释器，这就是推荐的单环境工作流。

可选 PT 参数：

- `--pt-python /path/to/python`
- `--pt-model-dir PianistTransformer/models/sft`
- `--pt-device auto|cuda|cpu`
- `--pt-temperature 1.0`
- `--pt-top-p 0.95`
- `--pt-max-tempo 300`

启用 PT 后，还会额外生成：

- `assets/output/expressive/raw/<stem>_pt_raw.mid`
- `assets/output/expressive/mapped/<stem>_pt_mapped.mid`

最终 WAV 默认使用 mapped 表情 MIDI 来渲染。

## 单独运行表情渲染

如果你已经有 cleanup 后的 MIDI，只想单独跑 PT 阶段：

```bash
python scripts/run_expressive_render.py \
  --input-midi assets/output/clean/hoshi_clean.mid \
  --output-root assets/output \
  --render \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

它会生成：

- `assets/output/expressive/raw/hoshi_pt_raw.mid`
- `assets/output/expressive/mapped/hoshi_pt_mapped.mid`
- `assets/output/rendered/hoshi_pt.wav`

## Linux 和 Slurm 说明

- 仓库面向 Linux
- baseline 各阶段不要求 GPU
- PianistTransformer 在有 GPU 时会更快
- 某些集群上，普通 shell 里 `torch.cuda.is_available()` 可能是 `False`，但进入 Slurm allocation 后会变成 `True`

如果你已经有一个交互式 GPU 作业，可以这样跑 PT：

```bash
srun --jobid=<your_jobid> --overlap bash -lc 'python scripts/run_expressive_render.py ...'
```

## 常见问题

### `basic-pitch` 因为 NumPy 兼容性报错

确保保持：

```bash
python -m pip install "numpy<2"
python -m pip install -r requirements.txt
```

### import 时 TensorFlow 打印大量 CUDA/XLA 警告

这是因为在统一的 Python `3.11` 环境里，`basic-pitch` 安装了 TensorFlow。日志会比较吵，但不一定代表 baseline 或 PT 阶段真的失败。

### 找不到 `basic-pitch`

确认环境激活：

```bash
conda activate pianist-transformer
which basic-pitch
```

### `fluidsynth` 或共享库缺失

请同时传：

```bash
--fluidsynth-bin "$FLUIDSYNTH_BIN" \
--fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

### PT 模型文件缺失

本地模型目录必须包含：

- `config.json`
- `generation_config.json`
- `model.safetensors`

路径是：

```text
PianistTransformer/models/sft/
```

## 最小验收清单

baseline：

```bash
conda activate pianist-transformer
python scripts/run_pipeline.py \
  --input sample.mp3 \
  --output-root assets/output \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

表情渲染：

```bash
python scripts/run_pipeline.py \
  --input sample.mp3 \
  --output-root assets/output \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR" \
  --enable-pianist-transformer
```

如果两条命令都能完成并生成预期文件，说明单环境工作流已经可用。
