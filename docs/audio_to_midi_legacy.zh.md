# 旧 audio-to-MIDI / PianistTransformer 流程

当前仓库主线是 `README.zh.md` 中的 AI cover 工作流。本页只记录旧的音频转 MIDI 和 PianistTransformer 表情渲染流程，方便需要复现实验或单独使用 MIDI pipeline 时参考。

## 环境

旧流程使用 `environment.yml`：

```bash
conda env create -f environment.yml
conda activate pianist-transformer
```

等价手动安装：

```bash
conda create -n pianist-transformer python=3.11 pip -y
conda activate pianist-transformer
python -m pip install -r requirements.txt
```

核心依赖包括：

- `basic-pitch`
- `pretty_midi`
- `midi2audio`
- `FluidSynth`
- `PianistTransformer`
- PyTorch `2.7.1` CUDA `11.8` wheel

## 流程

baseline 路径：

1. 音频预处理成统一 WAV。
2. 用 `basic-pitch` 转写成原始 MIDI。
3. 用 `pretty_midi` 清理过短音符。
4. 用 `midi2audio` + FluidSynth 渲染成 WAV。

可选表情渲染路径：

1. 完成 baseline 的预处理、转写和 MIDI cleanup。
2. 用 PianistTransformer 生成表情 raw MIDI。
3. 把表情 timing 映射回 score 对齐 MIDI。
4. 渲染 mapped MIDI 为最终 WAV。

默认输出：

```text
assets/output/convert/<stem>.wav
assets/output/raw/<stem>_basic_pitch.mid
assets/output/clean/<stem>_clean.mid
assets/output/expressive/raw/<stem>_pt_raw.mid
assets/output/expressive/mapped/<stem>_pt_mapped.mid
assets/output/rendered/<stem>.wav
```

## FluidSynth

旧流程同样需要 SoundFont 和 FluidSynth：

```bash
export SOUNDFONT=assets/soundfonts/extracted/usr/share/sounds/sf2/FluidR3_GM.sf2
export FLUIDSYNTH_BIN=assets/soundfonts/fluidsynth_pkg/usr/bin/fluidsynth
export FLUIDSYNTH_LIB_DIR=$(pwd)/assets/soundfonts/runtime_libs/usr/lib/x86_64-linux-gnu
```

检查：

```bash
LD_LIBRARY_PATH="$FLUIDSYNTH_LIB_DIR" "$FLUIDSYNTH_BIN" --version
```

## Baseline 命令

```bash
python scripts/run_pipeline.py \
  --input sample.mp3 \
  --output-root assets/output \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

预期输出：

```text
assets/output/convert/sample.wav
assets/output/raw/sample_basic_pitch.mid
assets/output/clean/sample_clean.mid
assets/output/rendered/sample.wav
```

## 启用 PianistTransformer

PianistTransformer 本地模型目录默认是：

```text
PianistTransformer/models/sft/
```

目录中至少需要：

```text
config.json
generation_config.json
model.safetensors
```

运行：

```bash
python scripts/run_pipeline.py \
  --input sample.mp3 \
  --output-root assets/output \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR" \
  --enable-pianist-transformer
```

常用参数：

- `--pt-python /path/to/python`
- `--pt-model-dir PianistTransformer/models/sft`
- `--pt-device auto|cuda|cpu`
- `--pt-temperature 1.0`
- `--pt-top-p 0.95`
- `--pt-max-tempo 300`

## 单独运行表情渲染

如果已经有 cleanup 后的 MIDI：

```bash
python scripts/run_expressive_render.py \
  --input-midi assets/output/clean/hoshi_clean.mid \
  --output-root assets/output \
  --render \
  --soundfont "$SOUNDFONT" \
  --fluidsynth-bin "$FLUIDSYNTH_BIN" \
  --fluidsynth-lib-dir "$FLUIDSYNTH_LIB_DIR"
```

输出：

```text
assets/output/expressive/raw/hoshi_pt_raw.mid
assets/output/expressive/mapped/hoshi_pt_mapped.mid
assets/output/rendered/hoshi_pt.wav
```

## 备注

- 旧 baseline 不要求 GPU。
- PianistTransformer 有 GPU 时更快。
- `basic-pitch` 依赖 `numpy<2`，不要随意升级到 NumPy 2。
- Python `3.11` 下 `basic-pitch` 可能安装 TensorFlow backend，import 时出现 TensorFlow CUDA/XLA warning 不一定代表失败。
