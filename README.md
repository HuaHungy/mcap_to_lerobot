# MCAP 到 LeRobot v2.1 

本项目提供将 ROS2 录制的 MCAP 数据集转换为 LeRobot v2.1 训练格式的工具，支持将相机图像合成为视频（MP4 容器，AV1 编码，yuv420p 像素格式，30 FPS），并生成与训练兼容的元数据与分层目录结构。

## 特性概览
- 统一采样频率 30 Hz，同步多模态数据
- 图像直接写入视频，最终转码为 AV1（`libaom-av1`）以便在 Linux 上播放
- 支持目标分辨率：`360p (640x360)` 与 `720p (1280x720)`
- 输出标准分层结构：`meta/`、`data/`、`videos/`
- 生成训练所需的 `info.json`、`episodes.jsonl`、`tasks.jsonl`、`episodes_stats.jsonl`

## 环境要求
- 操作系统：Linux（建议 Ubuntu 20.04+/22.04）
- Python：3.8 及以上
- 必备系统依赖：
  - `ffmpeg`（需包含 `libaom-av1` 编码器用于 AV1 输出）
  - 可选：`svt-av1`（若自行编译 FFmpeg 以启用 `libsvtav1`）

## 使用 Conda 管理环境（推荐）
```bash
# 创建并激活环境（Python 版本可按需调整）
conda create -n mcap_to_lerobot python=3.10 -y
conda activate mcap_to_lerobot

# 安装 FFmpeg（conda-forge 渠道带 libaom-av1）
conda install -c conda-forge ffmpeg -y
ffmpeg -hide_banner -encoders | grep -i av1  # 应出现 libaom-av1

# 安装常用 Python 依赖（conda 优先安装大件）
conda install -c conda-forge numpy pandas opencv tqdm psutil matplotlib openpyxl -y

# 安装 MCAP 相关（pip）
pip install mcap mcap-ros2-support

# 自检
python -c "import cv2, numpy, pandas, tqdm, psutil, openpyxl; print('Conda OK')"
python -c "import mcap; import mcap_ros2; print('MCAP OK')"
```

注意：如果 `ffmpeg -encoders` 中不显示 `libaom-av1`，请确认使用了 `conda-forge` 渠道并重新安装 FFmpeg：
`conda install -c conda-forge ffmpeg --force-reinstall -y`

## Python 依赖安装（非 Conda）
```bash
# 建议在虚拟环境中安装
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install mcap mcap-ros2-support numpy pandas opencv-python tqdm psutil matplotlib openpyxl
```

## 安装 FFmpeg（AV1 支持）
- 安装系统包：
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```
- 检查是否支持 AV1 编码器：
```bash
ffmpeg -hide_banner -encoders | grep -i av1
# 期望至少看到：libaom-av1
```

如需更快的 AV1 编码（`libsvtav1`），通常需要自行编译启用；本脚本默认使用 `libaom-av1`，无需额外编译即可使用。

## 运行示例
```bash
python /home/kemove/Downloads/mcap_to_lerobot/mcap_to_lerobot_v2_1_standard_converter_linux.py \
  --input /path/to/your.mcap \
  --output /home/kemove/Downloads/mcap_to_lerobot/test \
  --resolution 720p
```

可选参数：
- `--excel-config <path>` 指定 Excel 配置文件；不提供时脚本会自动从 MCAP 探测 Topic 配置
- `--no-excel` 忽略 Excel，强制自动探测 Topic
- `--max-duration <seconds>` 限制最大处理时长（秒），默认 0 表示处理完整文件
- `--resolution {360p,720p}` 指定目标分辨率，默认 `720p`

入口参数与流程参考：`mcap_to_lerobot_v2_1_standard_converter_linux.py:4059-4096`、`main()`

## 输出目录结构
转换成功后，`--output` 指定目录下将生成：
```
meta/
  info.json
  episodes.jsonl
  tasks.jsonl
  episodes_stats.jsonl
  camera.json          # 若源目录存在则复制；否则跳过

data/
  chunk-000/
    episode_000000.parquet

videos/
  chunk-000/
    observation.images.head_camera/episode_000000.mp4
    observation.images.left_hand_camera/episode_000000.mp4
    observation.images.right_hand_camera/episode_000000.mp4
```
- 视频容器：MP4（`mov,mp4`）
- 视频编码：AV1（`codec_name=av1`，`libaom-av1`）
- 像素格式：`yuv420p`
- 帧率：`30 FPS`
- 分辨率：与 `--resolution` 一致（默认 `1280x720`）

视频写入与转码逻辑参考：
- 写入占位编码并生成容器：`mcap_to_lerobot_v2_1_standard_converter_linux.py:1455-1474`、`create_mp4_videos()`（`1594-1629`）
- AV1 转码（含进度条）：`mcap_to_lerobot_v2_1_standard_converter_linux.py:1477-1552`

## 验证输出为 AV1
```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,pix_fmt,avg_frame_rate,height,width,profile \
  -show_entries format=format_name,probe_score \
  -of default=noprint_wrappers=1:nokey=1 \
  /home/kemove/Downloads/mcap_to_lerobot/test/videos/chunk-000/observation.images.head_camera/episode_000000.mp4
```
期望输出类似：
```
av1
yuv420p
30/1
1280
720
Main
mov,mp4,m4a,3gp,3g2,mj2
100
```

## 工作流程概览
- 读取 MCAP 并提取消息，优先使用 ROS `header.stamp` 时间戳，回退到 `log_time`（稳定对齐）
- 解码 `sensor_msgs/CompressedImage` 为帧（BGR），统一缩放到目标分辨率（`1204`）
- 逐帧写入占位视频，写完后用 FFmpeg 转码为 AV1（`1477-1552`）
- 构建训练数据行并生成 Parquet 与元数据（`generate_lerobot_v21_dataset`）

关键实现位置：
- 图像解码与缩放：`mcap_to_lerobot_v2_1_standard_converter_linux.py:1204`
- 视频写入与转码：`mcap_to_lerobot_v2_1_standard_converter_linux.py:1455-1474`、`1477-1552`、`1617-1629`、`3474-3491`
- 元数据生成：`mcap_to_lerobot_v2_1_standard_converter_linux.py:2104-2134`、`2211-2269`、`2363-2449`、`2451-2596`
- 命令行入口：`mcap_to_lerobot_v2_1_standard_converter_linux.py:4059-4121`

## 常见问题与排查
- 缺少 `ffmpeg` 或不含 AV1 编码器：
  - 安装：`sudo apt-get install ffmpeg`
  - 检查 AV1 编码器：`ffmpeg -hide_banner -encoders | grep -i av1`
- 使用 Conda 时 FFmpeg 优先级：
  - 确认路径使用的是 Conda 版本：`which ffmpeg` 应指向 Conda 环境目录
  - 如果仍使用系统 `/usr/bin/ffmpeg`，请在激活环境后重开终端或运行 `hash -r`
- 进度条不显示：确保 `tqdm` 已安装且 `ffprobe` 能返回源视频时长；否则转码仍会进行但不显示百分比
- Excel 配置未提供：脚本会自动从 MCAP 探测 Topic；如需强制忽略 Excel，使用 `--no-excel`
- 相机命名：当前默认输出键为 `observation.images.<camera>`，相机名为 `head_camera`、`left_hand_camera`、`right_hand_camera`；若需自定义命名，可在后续版本中添加 CLI 参数支持

## 性能优化建议（不改变编码方式）
- 默认使用 `libaom-av1` 并开启多线程：内部启用了 `-threads <CPU>` 与 `-cpu-used 8`
- 如需更快：提高 `cpu-used`（内部固定 8），或使用更高性能的 FFmpeg 构建（启用 `libsvtav1`）
- 如需更小体积：提高 `crf`（例如 32），会略微牺牲画质

## 许可与版权
- 使用需遵循原数据与依赖库的许可证要求
