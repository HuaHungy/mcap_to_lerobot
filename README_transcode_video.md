# transcode_video.py 使用说明

该脚本用于批量将视频文件（如 MP4/AVI/MKV/MOV/WebM 等）转码为兼容播放的 MP4 容器，默认使用 H.264/AAC/yuv420p，并支持可选强制帧率与目录批处理。

> 注意：转换器主流程已改为 AV1（libaom-av1）。本工具适用于对已有视频进行 H.264 转码，或者作为备用兼容方案。

## 支持格式
- 输入：`.mp4 .avi .mov .mkv .webm .mpg .mpeg .mts .m2ts`
- 输出：`MP4` 容器（视频 H.264、音频 AAC、像素格式 yuv420p）

## 环境依赖
- 需要安装 `ffmpeg`
- 需要 Python 3.8+，不依赖额外 Python 包（仅标准库）

## 快速使用
```bash
python /home/kemove/Downloads/mcap_to_lerobot/transcode_video.py <输入文件或目录>
```
- 如果输入是文件：在同目录生成输出；MP4 输入将生成临时文件并覆盖原名
- 如果输入是目录：会递归查找并转码所有匹配的视频文件

## 常用参数
```bash
python transcode_video.py INPUT [--out-dir OUT_DIR] [--crf CRF] [--preset PRESET] [--pix-fmt PIX_FMT] [--fps FPS] [--suffix SUFFIX] [--dry-run]
```
- `INPUT`：输入文件或目录路径
- `--out-dir`：目录模式输出根目录（默认在输入目录下创建 `transcoded/`）
- `--crf`：画质因子（默认 23；越小越清晰，范围约 18~28）
- `--preset`：编码速度（默认 `medium`；`ultrafast`~`veryslow`）
- `--pix-fmt`：像素格式（默认 `yuv420p`）
- `--fps`：强制帧率（例如 `30`）
- `--suffix`：输出文件后缀（默认 `_h264`）
- `--dry-run`：仅打印命令，不执行

示例：
```bash
# 将单个文件转码为 H.264/AAC/yuv420p，30 FPS
python transcode_video.py /path/to/input.mp4 --fps 30

# 批量转码整个目录，输出到指定目录
python transcode_video.py /path/to/videos --out-dir /path/to/output --crf 23 --preset medium

# 仅打印将执行的命令，不真正转码
python transcode_video.py /path/to/input.mp4 --dry-run
```

## 行为说明
- MP4 输入：将输出临时文件（`.__tmp__h264.mp4`），完成后覆盖原名实现原子替换
- AVI/MOV/MKV/WebM 等输入：生成同名 `.mp4` 文件在输出目录
- 如果转码失败，会打印错误码并保留原文件

## 与主转换器的关系
- 主转换器脚本 `mcap_to_lerobot_v2_1_standard_converter_linux.py` 默认输出 AV1/MP4；
- `transcode_video.py` 提供批量转码为 H.264 的工具，适合需要兼容旧设备或浏览器的场景；
- 两者互不依赖，可单独使用。

## 验证输出
```bash
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,pix_fmt,avg_frame_rate -of default=noprint_wrappers=1:nokey=1 /path/to/output.mp4
# 期望：codec_name=h264、pix_fmt=yuv420p、avg_frame_rate=如 30/1
```

## 常见问题
- 缺少 ffmpeg：
  - 安装：`sudo apt-get install ffmpeg` 或 `conda install -c conda-forge ffmpeg`
- 速度与体积调整：
  - 降低 `crf`（例如 20）提升画质与体积；提高 `crf`（如 26）减小体积
  - 调整 `preset` 为 `faster` 或 `fast` 提升速度，画质略降
- 覆盖行为：MP4 输入会覆盖原文件名，建议备份或使用 `--out-dir`

## 许可
- 按依赖与源代码许可证要求使用
