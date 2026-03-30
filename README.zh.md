# 指弹吉他音频转 MIDI MVP

英文版文档：`README.md`

这个仓库是一个面向 Linux 的轻量级 MVP，用来把单轨指弹吉他音频转换成：

1. 一个转写得到的 MIDI 文件
2. 一个钢琴音色渲染得到的 WAV 文件

当前流程刻意保持简单、明确、易复现：

1. 先把输入音频预处理成统一格式的 WAV
2. 用 `basic-pitch` 把 WAV 转成原始 MIDI
3. 用 `pretty_midi` 删除特别短的音符
4. 用 `midi2audio` + `FluidSynth` 把清理后的 MIDI 渲染成 WAV

当前仓库不包含：

- 训练代码
- MT3 集成
- 批处理
- 前端或 Web UI

## 当前状态

这个 MVP 已经在 Linux 环境下完成端到端验证，当前使用的核心组件是：

- Python `3.10`
- `basic-pitch`
- `pretty_midi`
- `midi2audio`
- `FluidSynth`
- 一个 GM `.sf2` soundfont

当前实现以 CPU 为主，不依赖 CUDA，也不依赖 GPU 专用库。即使服务器上有 A800，这一阶段也不需要用 GPU。

## 流程概览

输入：

- 单个音频文件路径，例如 `sample.mp3`

输出：

- 预处理后的 WAV：`assets/output/convert/<name>.wav`
- 原始 MIDI：`assets/output/raw/<name>_basic_pitch.mid`
- 清理后的 MIDI：`assets/output/clean/<name>_clean.mid`
- 最终渲染 WAV：`assets/output/rendered/<name>.wav`

主入口脚本是：

```bash
python scripts/run_pipeline.py ...
```

它会输出：

- `INFO` 日志
- 4 个阶段的 `tqdm` 进度条
- 最终生成文件路径

## 项目结构

```text
.
├── README.md
├── README.zh.md
├── environment.yml
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml
├── assets/
│   ├── input/
│   ├── output/
│   │   ├── clean/
│   │   ├── convert/
│   │   ├── raw/
│   │   └── rendered/
│   └── soundfonts/
├── scripts/
│   ├── run_cleanup_midi.py
│   ├── run_pipeline.py
│   ├── run_preprocess_audio.py
│   ├── run_render.py
│   └── run_transcription.py
├── src/
│   ├── __init__.py
│   ├── audio_preprocess.py
│   ├── midi_cleanup.py
│   ├── render.py
│   ├── transcription.py
│   └── utils.py
└── tests/
    └── test_smoke.py
```

目录说明：

- `assets/input/`：可选，用来存放输入音频
- `assets/output/convert/`：统一格式的单声道 `22050 Hz` WAV
- `assets/output/raw/`：`basic-pitch` 生成的原始 MIDI
- `assets/output/clean/`：清理过短音符后的 MIDI
- `assets/output/rendered/`：最终渲染得到的 WAV
- `assets/soundfonts/`：soundfont 以及本地 `FluidSynth` 运行时资源
- `scripts/`：用户直接运行的命令行脚本
- `src/`：各阶段的实现模块

## 环境搭建

### 1. 创建环境

推荐直接使用：

```bash
conda env create -f environment.yml
conda activate dl
```

如果你想手动创建，也可以：

```bash
conda create -n dl python=3.10 pip -y
conda activate dl
python -m pip install -r requirements.txt
```

说明：

- `environment.yml` 是推荐的复现入口
- `requirements.txt` 里把 `numpy` 限制为 `<2`
- 这么做是因为 `basic-pitch` 使用的 `tflite-runtime` 在 `numpy 2.x` 下可能报错

如果 `conda` 还没初始化，需要先 source：

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate dl
```

### 2. 验证 Python 依赖

```bash
python - <<'PY'
import librosa
import soundfile
import basic_pitch
import pretty_midi
import midi2audio
import numpy
print("python_deps_ok")
PY
```

## FluidSynth 和 SoundFont 准备

渲染阶段依赖两个东西：

1. `fluidsynth` 可执行文件
2. 一个 `.sf2` soundfont 文件

### 方案 A：直接使用仓库里已有的本地运行时

如果你本地已经准备好了这些文件，可以直接用：

- soundfont：
  `assets/soundfonts/extracted/usr/share/sounds/sf2/FluidR3_GM.sf2`
- FluidSynth 二进制：
  `assets/soundfonts/fluidsynth_pkg/usr/bin/fluidsynth`
- 运行时库目录：
  `assets/soundfonts/runtime_libs/usr/lib/x86_64-linux-gnu`

建议先导出环境变量：

```bash
export SOUNDFONT=assets/soundfonts/extracted/usr/share/sounds/sf2/FluidR3_GM.sf2
export FLUIDSYNTH_BIN=assets/soundfonts/fluidsynth_pkg/usr/bin/fluidsynth
export FLUIDSYNTH_LIB_DIR=$(pwd)/assets/soundfonts/runtime_libs/usr/lib/x86_64-linux-gnu
```

### 方案 B：无 sudo 重建本地运行时

如果这些文件不在仓库里，可以在 Ubuntu/Debian 风格系统上这样重建：

```bash
mkdir -p assets/soundfonts
cd assets/soundfonts

apt download fluid-soundfont-gm
apt download fluidsynth
apt download libfluidsynth3
apt download libsdl2-2.0-0
apt download libinstpatch-1.0-2
apt download libdecor-0-0

mkdir -p extracted fluidsynth_pkg runtime_libs

dpkg-deb -x fluid-soundfont-gm_*.deb extracted
dpkg-deb -x fluidsynth_*.deb fluidsynth_pkg

for pkg in libfluidsynth3_*.deb libsdl2-2.0-0_*.deb libinstpatch-1.0-2_*.deb libdecor-0-0_*.deb; do
  dpkg-deb -x "$pkg" runtime_libs
done

cd ../..
```

然后导出：

```bash
export SOUNDFONT=assets/soundfonts/extracted/usr/share/sounds/sf2/FluidR3_GM.sf2
export FLUIDSYNTH_BIN=assets/soundfonts/fluidsynth_pkg/usr/bin/fluidsynth
export FLUIDSYNTH_LIB_DIR=$(pwd)/assets/soundfonts/runtime_libs/usr/lib/x86_64-linux-gnu
```

### 3. 验证 FluidSynth 运行时

```bash
LD_LIBRARY_PATH="$FLUIDSYNTH_LIB_DIR" "$FLUIDSYNTH_BIN" --version
```

预期输出中应该包含：

```text
FluidSynth runtime version 2.x
```

## 快速开始

假设你已经：

- 激活了 `dl` 环境
- 设置了 `SOUNDFONT`、`FLUIDSYNTH_BIN`、`FLUIDSYNTH_LIB_DIR`
- 使用示例文件 `sample.mp3`

那么直接运行：

```bash
python scripts/run_pipeline.py \
  --input sample.mp3 \
  --output-root assets/output \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

预期生成：

- `assets/output/convert/sample.wav`
- `assets/output/raw/sample_basic_pitch.mid`
- `assets/output/clean/sample_clean.mid`
- `assets/output/rendered/sample.wav`

## 分阶段运行

### 1. 音频预处理

```bash
python scripts/run_preprocess_audio.py \
  --input sample.mp3 \
  --output assets/output/convert/sample.wav \
  --sr 22050
```

作用：

- 转单声道
- 重采样到 `22050 Hz`
- 写出标准 `.wav`

### 2. 转写

```bash
python scripts/run_transcription.py \
  --input assets/output/convert/sample.wav \
  --output-dir assets/output/raw
```

预期输出：

- `assets/output/raw/sample_basic_pitch.mid`

### 3. MIDI cleanup

```bash
python scripts/run_cleanup_midi.py \
  --input-midi assets/output/raw/sample_basic_pitch.mid \
  --output-midi assets/output/clean/sample_clean.mid \
  --min-note-duration 0.05
```

当前 cleanup 只做一件事：

- 删除持续时间小于阈值的音符

这一步也已经包含在 `scripts/run_pipeline.py` 里。

### 4. 渲染

```bash
python scripts/run_render.py \
  --input-midi assets/output/clean/sample_clean.mid \
  --output-wav assets/output/rendered/sample_clean.wav \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

## 一条命令跑完整流程

队友在环境配置好后，最应该直接跑的是这条：

```bash
python scripts/run_pipeline.py \
  --input sample.mp3 \
  --output-root assets/output \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

它会做：

1. `sample.mp3 -> assets/output/convert/sample.wav`
2. `sample.wav -> assets/output/raw/sample_basic_pitch.mid`
3. `sample_basic_pitch.mid -> assets/output/clean/sample_clean.mid`
4. `sample_clean.mid -> assets/output/rendered/sample.wav`

运行时会显示：

- 阶段日志
- `tqdm` 进度条
- 最终生成文件路径

## 仓库里的示例音频

第一次建议直接用你自己的输入音频路径，例如：

```bash
python scripts/run_pipeline.py \
  --input /path/to/your/input.mp3 \
  --output-root assets/output \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

## 配置文件说明

仓库里有一个占位配置文件：

```text
configs/default.yaml
```

当前脚本主要还是命令行参数驱动，`default.yaml` 目前更像是路径和默认值参考。

当前包含的 key：

- `input_audio`
- `preprocess_output_dir`
- `preprocessed_audio`
- `transcription_output_dir`
- `cleaned_midi_output_dir`
- `render_output_dir`
- `output_dir`
- `soundfont_path`
- `min_note_duration`
- `preprocess_sample_rate`
- `preprocess_mono`

## Linux 说明

- 当前 MVP 面向 Linux
- 当前实现不依赖 CUDA
- 当前实现不使用 GPU
- 渲染依赖 `FluidSynth` 和有效的 `.sf2`
- 如果你不用系统安装的 `FluidSynth`，需要同时传 `--fluidsynth-lib-dir`

## 已知限制

- 只支持单文件处理
- 没有 batch 模式
- 没有 MT3
- 没有前端
- 没有训练代码
- cleanup 目前只删除过短音符
- 渲染使用通用 GM soundfont，不是专门的吉他音色模型

## 常见问题

### `basic-pitch` 报 NumPy / TFLite 错误

执行：

```bash
python -m pip install "numpy<2"
python -m pip install -r requirements.txt
```

### 找不到 `basic-pitch`

确认环境激活：

```bash
conda activate dl
which basic-pitch
```

### 找不到 `fluidsynth`

两种办法：

- 系统级安装 `fluidsynth`
- 用本文档里的本地运行时方案，并传 `--fluidsynth-bin`

### 找不到 soundfont

检查：

```bash
ls "$SOUNDFONT"
```

### 渲染时共享库报错

如果你使用的是仓库里的本地 `FluidSynth` 运行时，请确认传了：

```bash
--fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

并且它指向：

```text
assets/soundfonts/runtime_libs/usr/lib/x86_64-linux-gnu
```

### 流程跑得慢

这是当前 MVP 的预期表现：

- 转写走 CPU
- MIDI 渲染对长音频也会花时间

当前优先级是简单和可复现，不是速度。

## 最小验收清单

队友完成环境配置后，应该能够运行：

```bash
conda activate dl
python scripts/run_pipeline.py \
  --input sample.mp3 \
  --output-root assets/output \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

然后确认这四个文件存在：

```bash
ls assets/output/convert/sample.wav
ls assets/output/raw/sample_basic_pitch.mid
ls assets/output/clean/sample_clean.mid
ls assets/output/rendered/sample.wav
```

如果这四个文件都存在，说明当前 MVP 已经端到端跑通。
