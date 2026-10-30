#!/usr/bin/env python3
"""Create synchronized P0/P1/P2 MuJoCo comparison videos.

The three source MP4 files are opened read-only.  All generated files are
written to the comparison directory beside this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 510
PANEL_WIDTH = 640
PANEL_HEIGHT = 360
VIDEO_Y = 150
FPS = 30


def run(command: list[str]) -> None:
    # Keep console output CP932-safe on Japanese Windows.  The full Unicode
    # filter text is still passed directly to FFmpeg as a subprocess argument.
    print(f"Running: {Path(command[0]).name} -> {command[-1]}", flush=True)
    subprocess.run(command, check=True)


def probe(path: Path, ffprobe: str) -> dict:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,format_name:stream=index,codec_name,codec_type,width,height,"
            "r_frame_rate,avg_frame_rate,nb_frames,pix_fmt",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_stream(metadata: dict) -> dict:
    return next(stream for stream in metadata["streams"] if stream["codec_type"] == "video")


def validate_source(path: Path, metadata: dict) -> None:
    stream = video_stream(metadata)
    frame_rate = float(Fraction(stream["avg_frame_rate"]))
    duration = float(metadata["format"]["duration"])
    problems: list[str] = []
    if (stream["width"], stream["height"]) != (1280, 720):
        problems.append(f"resolution={stream['width']}x{stream['height']}")
    if abs(frame_rate - FPS) > 1e-6:
        problems.append(f"fps={frame_rate}")
    if duration < 30.0:
        problems.append(f"duration={duration:.3f}s")
    if problems:
        raise RuntimeError(f"Unexpected source metadata for {path}: {', '.join(problems)}")


def create_header(output: Path, font_regular: Path, font_bold: Path) -> None:
    image = Image.new("RGB", (CANVAS_WIDTH, VIDEO_Y), (16, 21, 27))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, CANVAS_WIDTH, 57), fill=(5, 8, 12))
    draw.rectangle((0, 58, 639, VIDEO_Y - 1), fill=(32, 42, 54))
    draw.rectangle((640, 58, 1279, VIDEO_Y - 1), fill=(24, 33, 43))
    draw.rectangle((1280, 58, 1919, VIDEO_Y - 1), fill=(32, 42, 54))

    title_font = ImageFont.truetype(str(font_bold), 32)
    name_font = ImageFont.truetype(str(font_bold), 28)
    label_font = ImageFont.truetype(str(font_regular), 22)

    def centered(text: str, center_x: float, y: int, font: ImageFont.FreeTypeFont) -> None:
        bounds = draw.textbbox((0, 0), text, font=font)
        width = bounds[2] - bounds[0]
        draw.text((center_x - width / 2, y), text, font=font, fill="white")

    centered("MuJoCo Walking Comparison — Command Speed: 1.90 m/s", 960, 7, title_font)
    for index, (name, label) in enumerate(
        [
            ("P0", "Contact tolerance: 0 ms"),
            ("P1", "Contact tolerance: ±20 ms"),
            ("P2", "Contact tolerance: ±40 ms"),
        ]
    ):
        center_x = index * PANEL_WIDTH + PANEL_WIDTH / 2
        centered(name, center_x, 61, name_font)
        centered(label, center_x, 105, label_font)
    image.save(output)


def create_header_track(ffmpeg: str, header: Path, output: Path, frame_count: int) -> None:
    image = Image.open(header).convert("RGB")
    if image.size != (CANVAS_WIDTH, VIDEO_Y):
        raise RuntimeError(f"Unexpected header size: {image.size}")
    frame_bytes = image.tobytes()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{CANVAS_WIDTH}x{VIDEO_Y}",
        "-framerate",
        str(FPS),
        "-i",
        "pipe:0",
        "-frames:v",
        str(frame_count),
        "-c:v",
        "ffv1",
        "-level",
        "3",
        str(output),
    ]
    print(f"Running: {Path(ffmpeg).name} -> {output}", flush=True)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for _ in range(frame_count):
            process.stdin.write(frame_bytes)
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def build_filter() -> str:
    filters = [
        f"[0:v]setpts=PTS-STARTPTS,fps={FPS},scale={PANEL_WIDTH}:{PANEL_HEIGHT}:flags=lanczos,setsar=1[p0]",
        f"[1:v]setpts=PTS-STARTPTS,fps={FPS},scale={PANEL_WIDTH}:{PANEL_HEIGHT}:flags=lanczos,setsar=1[p1]",
        f"[2:v]setpts=PTS-STARTPTS,fps={FPS},scale={PANEL_WIDTH}:{PANEL_HEIGHT}:flags=lanczos,setsar=1[p2]",
        "[p0][p1][p2]hstack=inputs=3[stack]",
        f"[3:v]setpts=PTS-STARTPTS,fps={FPS},scale={CANVAS_WIDTH}:{VIDEO_Y},setsar=1[header]",
        "[header][stack]vstack=inputs=2:shortest=1[outv]",
    ]
    return ";".join(filters)


def encode(
    ffmpeg: str,
    sources: list[Path],
    header_track: Path,
    output: Path,
    filter_complex: str,
    duration: float | None,
) -> None:
    command = [ffmpeg, "-hide_banner", "-y"]
    for source in sources:
        command.extend(["-i", str(source)])
    command.extend(["-i", str(header_track)])
    command.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            # Use independent H.264 frames.  This avoids intermittent damage
            # to the static header observed with this Windows FFmpeg build's
            # long-GOP encoding while keeping the required H.264 format.
            "-g",
            "1",
            "-keyint_min",
            "1",
            "-sc_threshold",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(FPS),
            "-movflags",
            "+faststart",
        ]
    )
    if duration is not None:
        command.extend(["-t", f"{duration:g}"])
    command.append(str(output))
    run(command)


def save_preview(ffmpeg: str, video: Path, time_s: float, output: Path) -> None:
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-y",
            "-i",
            str(video),
            "-ss",
            f"{time_s:g}",
            "-frames:v",
            "1",
            "-update",
            "1",
            str(output),
        ]
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir.parent,
        help="Directory containing P0/P1/P2 source MP4 files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Directory for generated videos, previews, and manifest",
    )
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    p1_candidates = [input_dir / "P1_v190_30s.mp4", input_dir / "P1_v190_30s(1).mp4"]
    p1 = next((path for path in p1_candidates if path.is_file()), None)
    if p1 is None:
        raise FileNotFoundError("Neither P1_v190_30s.mp4 nor P1_v190_30s(1).mp4 was found")

    sources = [input_dir / "P0_v190_30s.mp4", p1, input_dir / "P2_v190_30s.mp4"]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    source_records = []
    for source in sources:
        metadata = probe(source, ffprobe)
        validate_source(source, metadata)
        source_records.append(
            {
                "path": str(source),
                "sha256": sha256(source),
                "metadata": metadata,
            }
        )

    frame_count = min(
        int(video_stream(record["metadata"])["nb_frames"]) for record in source_records
    )

    font_regular = Path(r"C:\Windows\Fonts\arial.ttf")
    font_bold = Path(r"C:\Windows\Fonts\arialbd.ttf")
    if not font_regular.is_file() or not font_bold.is_file():
        raise FileNotFoundError("Required Arial font files were not found")

    header = output_dir / "comparison_header.png"
    create_header(header, font_regular, font_bold)
    header_track = output_dir / "comparison_header_track.mkv"
    create_header_track(ffmpeg, header, header_track, frame_count)
    filter_complex = build_filter()
    output_30 = output_dir / "MuJoCo_P012_1p90_30s_3panel.mp4"
    output_20 = output_dir / "MuJoCo_P012_1p90_20s_3panel.mp4"
    encode(ffmpeg, sources, header_track, output_30, filter_complex, duration=None)
    encode(ffmpeg, sources, header_track, output_20, filter_complex, duration=20.0)

    previews = [
        (0.0, output_dir / "preview_00s.png"),
        (14.5, output_dir / "preview_14p5s.png"),
        (29.5, output_dir / "preview_29p5s.png"),
    ]
    for time_s, preview in previews:
        save_preview(ffmpeg, output_30, time_s, preview)

    report = {
        "sources": source_records,
        "outputs": [
            {
                "path": str(output_30),
                "metadata": probe(output_30, ffprobe),
            },
            {
                "path": str(output_20),
                "metadata": probe(output_20, ffprobe),
            },
        ],
        "layout": {
            "canvas": [CANVAS_WIDTH, CANVAS_HEIGHT],
            "panel": [PANEL_WIDTH, PANEL_HEIGHT],
            "video_y": VIDEO_Y,
            "fps": FPS,
            "panel_order": ["P0", "P1", "P2"],
            "header_path": str(header),
            "header_track_path": str(header_track),
            "header_track_frame_count": frame_count,
        },
        "encoding": {
            "video_codec": "H.264 (libx264)",
            "pixel_format": "yuv420p",
            "crf": 18,
            "all_intra": True,
            "audio": False,
        },
        "previews": [{"time_s": time_s, "path": str(path)} for time_s, path in previews],
    }
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Completed successfully.")


if __name__ == "__main__":
    main()
