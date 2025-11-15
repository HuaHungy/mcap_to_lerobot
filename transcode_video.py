#!/usr/bin/env python3
import argparse
import subprocess
import shutil
import sys
from pathlib import Path
from typing import List

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".mts", ".m2ts"}

def normalize_path(p: str) -> Path:
    p = p.replace("\\", "/")
    return Path(p).expanduser().resolve()

def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print("错误：未找到 ffmpeg，请先安装：sudo apt-get install ffmpeg")
        sys.exit(1)

def build_output_path(input_path: Path, out_dir: Path | None, suffix: str = "_h264") -> Path:
    stem = input_path.stem
    ext = input_path.suffix.lower()
    if out_dir is None:
        out_dir = input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    # MP4：输出到临时文件，稍后用原名覆盖
    if ext == ".mp4":
        return out_dir / f"{stem}.__tmp__h264.mp4"
    # 其他容器（如 AVI）：同名 .mp4
    return out_dir / f"{stem}.mp4"

def list_videos_under(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            files.append(p)
    return files

def transcode_file(
    input_path: Path,
    output_path: Path,
    crf: int = 23,
    preset: str = "medium",
    pix_fmt: str = "yuv420p",
    fps: int | None = None,
    dry_run: bool = False,
) -> bool:
    # ffmpeg 参数：视频 libx264，yuv420p，音频 aac（可选），faststart
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(input_path),
        "-map", "0:v:0",
        "-map", "0:a?:0",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", pix_fmt,
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
    ]
    if fps:
        cmd.extend(["-r", str(fps)])
    cmd.append(str(output_path))

    print(f"转码: {input_path} -> {output_path}")
    if dry_run:
        print("DRY-RUN 命令：", " ".join(cmd))
        return True

    try:
        res = subprocess.run(cmd, check=False)
        if res.returncode == 0:
            # MP4 输入：用原子替换覆盖原文件名
            if input_path.suffix.lower() == ".mp4":
                try:
                    output_path.replace(input_path)
                    print(f"完成并覆盖原文件: {input_path}")
                except Exception as e:
                    print(f"覆盖原文件失败: {input_path} -> {e}")
                    return False
            else:
                print(f"完成: {output_path}")
            return True
        else:
            print(f"失败({res.returncode}): {input_path}")
            return False
    except Exception as e:
        print(f"异常: {input_path} -> {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="批量转码为 H.264/AAC/yuv420p 的 MP4")
    parser.add_argument("input", help="输入文件或目录")
    parser.add_argument("--out-dir", help="目录模式下的输出根目录，默认在输入目录下创建 transcoded/")
    parser.add_argument("--crf", type=int, default=23, help="x264 质量(小=好，范围 18~28)")
    parser.add_argument("--preset", default="medium", help="x264 速度(ultrafast...veryslow)")
    parser.add_argument("--pix-fmt", default="yuv420p", help="像素格式，默认 yuv420p")
    parser.add_argument("--fps", type=int, default=None, help="可选强制帧率，例如 30")
    parser.add_argument("--suffix", default="_h264", help="输出文件后缀，默认 _h264")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不执行")
    args = parser.parse_args()

    ensure_ffmpeg()

    in_path = normalize_path(args.input)

    if not in_path.exists():
        print(f"错误：输入不存在：{in_path}")
        sys.exit(1)

    out_root = normalize_path(args.out_dir) if args.out_dir else None

    if in_path.is_file():
        output_path = build_output_path(in_path, out_root, args.suffix)
        ok = transcode_file(
            in_path, output_path,
            crf=args.crf, preset=args.preset,
            pix_fmt=args.pix_fmt, fps=args.fps,
            dry_run=args.dry_run,
        )
        sys.exit(0 if ok else 2)
    else:
        # 目录模式
        videos = list_videos_under(in_path)
        if not videos:
            print(f"提示：目录下未发现可识别视频：{in_path}")
            sys.exit(0)

        if out_root is None:
            out_root = in_path / "transcoded"
        out_root.mkdir(parents=True, exist_ok=True)

        total = len(videos)
        print(f"发现 {total} 个视频，输出到：{out_root}")
        ok_count = 0
        for i, v in enumerate(videos, start=1):
            rel = v.relative_to(in_path)
            target_dir = (out_root / rel.parent)
            output_path = build_output_path(v, target_dir, args.suffix)
            target_dir.mkdir(parents=True, exist_ok=True)
            if transcode_file(
                v, output_path,
                crf=args.crf, preset=args.preset,
                pix_fmt=args.pix_fmt, fps=args.fps,
                dry_run=args.dry_run,
            ):
                ok_count += 1
            print(f"[{i}/{total}] 完成数：{ok_count}")
        print(f"全部完成：成功 {ok_count}/{total}")
        sys.exit(0 if ok_count == total else 3)

if __name__ == "__main__":
    main()