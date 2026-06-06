# Backend Inference Optimization Analysis

本文档基于 git 历史记录（`df74b83` 到 `f150c45`），分析后端 AI Cover 推理流水线中的所有性能优化。

---

## Overview

后端推理流水线主要包含以下阶段：

1. **Stage 0**: 音频预处理（重采样、移调）
2. **Stage 1**: Demucs 人声分离
3. **Stage 2**: DDSP-SVC 音色转换
4. **Stage 3**: Pop2Piano 钢琴伴奏生成
5. **Stage 4**: MIDI 到 WAV 渲染（FluidSynth）
6. **Stage 5**: 最终混音

所有优化按时间顺序排列如下：

---

## 1. 记录阶段耗时 (Commit `c4eff21`)

**日期**: 2026-06-01

**改动文件**:
- `src/ai_cover.py`
- `backend/jobs.py`
- `third_party/ai_cover/DDSP-SVC/main_reflow.py`
- `scripts/run_ai_piano_cover.py`
- `scripts/run_pop2piano.py`

**优化内容**:
- 在每个推理阶段前后插入 `time.perf_counter()` 计时点
- `run_ai_piano_cover` 返回值从 `CoverArtifacts` 扩展为 `CoverRunResult`（包含 `stage_timings` 字典）
- 在 DDSP-SVC `main_reflow.py` 的 segment 循环中添加细粒度计时：`units_encode`、`model_forward`、`vocoder_infer` 拆分为独立指标
- 在 `run_pop2piano.py` 中记录 `load_model`、`load_audio`、`preprocess`、`model_generate`、`decode`、`save_midi` 等子阶段耗时
- 完成后输出 `timings.json` 文件，便于性能回归监控

**效果**: 为所有后续优化提供了定量基线，可精确衡量每个优化的收益。

---

## 2. 权重预加载 (Commit `c4eff21`, 同 commit)

**改动文件**:
- `backend/model_registry.py`

**优化内容**:
- 扩展 `preload_mode: torch_cpu` 模式：启动时将 Demucs Separator、Pop2Piano Processor/Model、RMVPE pitch extractor、DDSP checkpoint 权重（含 encoder/vocoder）**全部预加载到内存**
- 新增 `loaded_components` 字典管理预加载的运行时组件
- 每个角色记录 `preload_seconds` 指标
- 新增 `preload_mode: torch_cuda` 模式（后续 commit 扩展）：直接在 CUDA 上预加载，跳过子进程启动

**效果**: 消除首次推理任务时的模型加载延迟，实际请求只需执行推理计算，无需等待 IO 和权重初始化。

---

## 3. Reflow 模型 Conditioner 缓存 (Commit `8c2a9bf`)

**日期**: 2026-06-01

**改动文件**:
- `third_party/ai_cover/DDSP-SVC/reflow/lynxnet.py`
- `third_party/ai_cover/DDSP-SVC/reflow/reflow.py`

**优化内容**:
- **问题**: DDSP-SVC 的 Reflow（RectifiedFlow）ODE 求解器在每个时间步调用 `LYNXNet.forward()`，每次都执行 `conditioner_projection(cond)`。对于同一段音频的同一个 segment，conditioner 在整个 ODE 求解过程中不变（RK4 需要 4 次 velocity_fn 调用，Euler 需要 `infer_step` 次），但每次都重复投影，浪费大量计算。
- **方案**:
  - `LYNXNet` 新增 `prepare_conditioner(cond)` 方法：一次性对所有 residual layer 执行 `conditioner_projection`，返回投影结果列表
  - `LYNXNet.forward_with_cond_cache()` 接收预计算的投影缓存，跳过重复投影
  - `RectifiedFlow._prepare_cond_cache()` / `_velocity()`: 在 ODE 求解开始前调用一次 `prepare_conditioner`，后续所有步骤复用缓存
  - `LYNXNetResidualLayer.forward_with_conditioner_projection()`: 直接接收投影结果，避免重复计算

**效果**: 对于 infer_step=20 的 Euler 求解，**conditioner 投影从每个 segment 的 20 次减少为 1 次**；RK4 从 4 次减少为 1 次。DDSP forward 主循环计算量显著降低。

---

## 4. 用 beat-this 替代 essentia (Commit `b7ce0ea`)

**日期**: 2026-06-01

**改动文件**:
- `pop2piano/preprocess/beat_quantizer.py`
- `pop2piano/preprocess/bpm_quantize.py`
- `pop2piano/transformer_wrapper.py`
- `environment.yml`, `requirements.txt` 等依赖文件

**优化内容**:
- **问题**: 原 Pop2Piano 使用 `essentia`（C++ 库）做节拍检测，只能 CPU 运行
- **方案**: 替换为 `beat-this`（PyTorch-based 神经网络节拍检测器）
  - `extract_rhythm()`: 用 `File2Beats` 替代 `essentia.standard.RhythmExtractor2013`
  - 新增 `get_file_beat_tracker()` 懒加载全局单例，避免重复初始化
  - `pop2piano/transformer_wrapper.py` 传入 `device` 参数给 `extract_rhythm()`
  - 依赖更新: `essentia==2.1b6.dev1034` → `beat-this==1.1.0`, `torch>=2`, `torchaudio>=2`

**效果**: 节拍提取从 CPU 单核计算迁移到 GPU（CUDA）加速，充分利用 PyTorch 推理框架。同时消除了 `essentia` 复杂的 C++ 依赖。

---

## 5. 启动预热阶段 (Commit `26435b7`)

**日期**: 2026-06-02

**改动文件**:
- `backend/main.py`
- `backend/model_registry.py`
- `backend/settings.py`
- `src/ai_cover.py`
- `configs/roles.yaml`

**优化内容**:
- 后端启动后自动跑一次完整 cover pipeline（使用 `warmup.mp3`），输出到 `assets/output/jobs/_startup_warmup`
- 配置项:
  - `warmup_enabled`: 是否启用（默认 true）
  - `warmup_audio`: 预热用的输入文件
  - `warmup_pre_pitch_shift: 1.0`: 非默认移调值模拟真实请求
  - `warmup_vocals_volume: 0.9`, `warmup_piano_volume: 1.1`: 非默认混音参数
- `run_ai_piano_cover()` 新增 `pop2piano_beat_checkpoint` 参数支持
- Beat-This beat tracker 随 Pop2Piano 一起预加载
- 预热完成后打印 `startup complete`

**效果**: 首次推理的 CUDA kernel 编译和 GPU 内存分配在预热阶段完成，用户实际请求获得的是"热"状态下的性能。解决 CUDA JIT 编译导致的首次推理延迟。

---

## 6. Stage 1+2 与 Stage 3+4 并行化 (Commit `6cc60b4`)

**日期**: 2026-06-03

**改动文件**:
- `src/ai_cover.py`（主要改动）
- `src/ddsp_inprocess.py`
- `backend/jobs.py`, `backend/settings.py`
- `scripts/run_ai_piano_cover.py`

**优化内容**:
- **核心思想**: Stage 1+2（人声分离 + 音色转换）与 Stage 3+4（钢琴生成 + MIDI 渲染）之间没有数据依赖，可以并行执行
- **实现细节**:
  - `run_vocal_branch()` / `run_piano_branch()`: 将两条分支封装为独立闭包
  - 使用 `ThreadPoolExecutor(max_workers=2)` 并行提交两条分支
  - **CUDA Stream 隔离**: 每条分支使用独立的 `torch.cuda.Stream`，避免竞争
    - `_cuda_stream_context(device, branch_name)`: 为每个分支创建/复用专用 stream
    - `_get_cuda_stream()`: 线程安全的 stream 缓存
    - 分支执行完成后 `stream.synchronize()`
  - **设备管理**: `_cuda_branch_device()` 自动检测每个分支所需的 CUDA 设备
  - **CUDA 内存清理**: 每个分支完成后或异常时调用 `_cleanup_cuda_memory()`
  - `USE_PARALLEL_COVER_STAGES = True` 全局开关，支持 `--no-parallel-stages` 降级为串行
  - Demucs separator 增加 `_move_demucs_separator()` 自动设备迁移

- 进度回调优化: `_handle_progress` 改为 max-aggregation，确保并行分支进度报告不会倒退

- DDSP in-process 相关的独立优化:
  - `torch.no_grad()` → `torch.inference_mode()` (更高效，完全禁用 autograd tracking)
  - F0 extractor 复用预加载实例，避免重复创建
  - Demucs 分离后不再计算 no_vocals（去掉不必要的伴奏合并），减少显存和计算
  - DDSP 各子模块支持设备迁移 (`_move_units_encoder`, `_move_ddsp_vocoder` 等)

**效果**: 理想情况下总延迟 = max(Stage1+2, Stage3+4) + Stage5，而非之前的累加。在 CUDA stream 级别实现并行，充分利用 GPU 多流并发能力。

---

## 7. DDSP Segment 批量推理 (Commit `ca25eae`)

**日期**: 2026-06-03

**改动文件**:
- `src/ddsp_inprocess.py`（主要改动）
- `src/ai_cover.py`
- `backend/jobs.py`, `backend/model_registry.py`, `backend/settings.py`
- `configs/roles.yaml`

**优化内容**:
- **问题**: DDSP-SVC 的 segment 循环逐个处理音频切片，每个 segment 单独调用 `model()` 和 `vocoder.infer()`。GPU kernel launch overhead 和 host-device 同步开销累积很大。
- **方案**: 将多个 segment 组合为一个 batch 进行批量推理
  - 新增 `_EncodedSegment` / `_SegmentMeta` / `_RenderedBatch` 数据结构
  - `_length_sorted_batches()`: 按 segment 帧长排序后分组，减少 padding 开销
  - `pad_time_batch()`: 将不等长 segment 的 units/f0/volume pad 到统一长度
  - `render_segment_batch()`: 批量执行 model forward + vocoder infer
  - `batch_mask_for_output()`: 为每个 segment 构造正确的 mask
  - **回退策略**: 如果批量推理 OOM（`RuntimeError`），自动回退为逐个 segment 推理
  - **Attention 兼容**: 若 `model.use_attention=True`，自动禁用 batching（当前模型无 padding mask）
- 可配置项:
  - `ddsp_segment_batch_size`: 每个 role 可配置（默认 4），支持环境变量 `DDSP_SEGMENT_BATCH_SIZE` 覆盖
  - `configs/roles.yaml` 每个角色可独立设置

- 输出组装优化:
  - 所有 segment 的 GPU 输出一次性转移到 CPU（合并 CPU transfer）
  - 组装逻辑先收集再 concat（避免反复 `np.append`）

**效果**: 将 N 次小 batch 的 GPU kernel launch 合并为 N/batch_size 次大 batch launch，显著减少 host-device 同步和 kernel launch overhead。接近线性地减少了 seg_loop 总耗时中的 overhead 部分。

---

## 8. 前端预上传音频 (Commit `43e5400`)

**日期**: 2026-06-03

**改动文件**:
- `backend/main.py`
- `dashboard/src/App.tsx`

**优化内容**:
- **问题**: 用户选择文件 → 点击提交 → 上传音频 → 开始推理。上传和推理串行，用户等待时间长。
- **方案**: 前端在选择音频文件后**立即上传**到后端
  - 新增 `POST /api/uploads` 接口：接收音频文件，返回 `upload_id`
  - 新增 `DELETE /api/uploads/{upload_id}` 接口：取消未使用的上传
  - 音频存储在 `_pending` 临时目录
  - 提交任务时传 `upload_id` 而非 `audio` 文件：`POST /api/jobs` 改为支持 `upload_id` 或 `audio` 二选一
  - `_claim_uploaded_audio()`: 从 pending 目录移动文件到 job 目录
- 前端逻辑:
  - `beginAudioUpload(file)`: 文件选择后立即异步上传，使用 `AbortController` 支持取消
  - 上传状态追踪: `uploadingAudio` / `uploadedAudio` 状态变量
  - UI 显示 Badge: "Uploading" → "Uploaded" → "Ready"
  - 提交时直接使用已上传的 `upload_id`，无需等待文件传输

**效果**: 将网络传输延迟从"用户点击提交 → 开始推理"提前到"用户选择文件 → 调参"阶段，**用户感知等待时间减少约等于上传时间**。

---

## 9. Stage 0 用 PyTorch 音频处理替代 FFmpeg (Commit `f150c45`)

**日期**: 2026-06-03

**改动文件**:
- `src/ai_cover.py`

**优化内容**:
- **问题**: Stage 0 预处理使用 FFmpeg + libRubberBand 做 pitch shift，需要子进程启动 + CPU 信号处理
- **方案**: 新增 `preprocess_pitch_shift_torchaudio_cuda()`
  - 使用 `torchaudio.transforms.PitchShift` 在 CUDA 上执行移调
  - `torchaudio.load()` 直接加载音频为 tensor
  - 自动 fallback: 如果 torchaudio 无法解析格式（如 mp3/flac），用 FFmpeg 解码为 WAV 再加载
  - `torch.inference_mode()` 下执行，无 autograd overhead
  - `torch.cuda.synchronize()` 确保 GPU 操作完成后转移到 CPU
  - `torchaudio.save()` 以 PCM_F（32-bit float）格式输出，比之前的 s16le 精度更高
- 全局配置 `PREPROCESS_PITCH_SHIFT_METHOD = "torchaudio-pitchshift-cuda"`
- 移调值为 0 时跳过整个预处理阶段

**效果**: 
- 移调操作从 FFmpeg CPU 子进程迁移到 PyTorch CUDA，利用 GPU 并行计算
- 消除子进程启动开销
- 输出格式从 16-bit 整数升级为 32-bit 浮点，减少后续阶段的量化损失

---

## Summary of All Optimizations

| # | 优化类别 | Commit | 核心思路 |
|---|---------|--------|---------|
| 1 | **可观测性** | `c4eff21` | 全流程细粒度计时，建立性能基线 |
| 2 | **模型预加载** | `c4eff21` | 启动时加载所有权重到内存/CUDA，消除首次推理 IO 延迟 |
| 3 | **计算消除** | `8c2a9bf` | Reflow ODE 求解中缓存 conditioner 投影，每个 segment 减少 N-1 次投影 |
| 4 | **GPU 迁移** | `b7ce0ea` | 节拍检测从 CPU (essentia) 迁移到 GPU (beat-this/PyTorch) |
| 5 | **预热** | `26435b7` | 启动后预热运行一次完整推理，消除 CUDA JIT 编译延迟 |
| 6 | **并行化** | `6cc60b4` | Stage1+2 与 Stage3+4 通过 CUDA Stream 并行执行 |
| 7 | **批量计算** | `ca25eae` | DDSP segment 批量推理，减少 GPU kernel launch overhead |
| 8 | **前端预上传** | `43e5400` | 文件选择时立即上传，网络传输与参数设置并行 |
| 9 | **GPU 音频处理** | `f150c45` | Stage0 移调从 FFmpeg CPU 迁移到 torchaudio CUDA |

**整体效果**: 通过**并行化**（#6）、**批量计算**（#7）、**计算消除**（#3）、**GPU 加速**（#4, #9）、**预加载与预热**（#2, #5）以及**前端预上传**（#8），端到端推理延迟从最初的数分钟级别优化到目标数十秒级别。
