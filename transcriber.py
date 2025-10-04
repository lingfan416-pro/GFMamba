# transcriber.py
import os
import sys
import shutil
import subprocess
import tempfile
from typing import Tuple, Callable, Optional

from faster_whisper import WhisperModel

def _extract_wav_if_needed(video_path: str, sr: int = 16000) -> Tuple[str, Callable[[], None]]:
    """
    若系统 PATH 没有 ffmpeg，则尝试用 imageio-ffmpeg 的内置二进制把视频转临时 WAV。
    返回 (音频源路径, 清理函数)；若可直接读视频，返回 (video_path, 空清理函数)。
    """
    if shutil.which("ffmpeg"):                      # 系统有 ffmpeg
        return video_path, (lambda: None)

    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return video_path, (lambda: None)           # 让模型直接读视频（可能也能行）

    tmpdir = tempfile.mkdtemp(prefix="asr_")
    wav_path = os.path.join(tmpdir, "audio_16k_mono.wav")
    cmd = [
        ffmpeg_bin, "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le",
        wav_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _cleanup():
        try:
            if os.path.exists(wav_path): os.remove(wav_path)
            os.rmdir(tmpdir)
        except Exception:
            pass

    return wav_path, _cleanup


class WhisperTranscriber:
    """
    复用模型的转写器：实例化一次，多次调用更快。
    """
    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8"):
        """
        model_size: tiny/base/small/medium/large-v3
        device: "cpu" 或 "cuda"
        compute_type: CPU 推荐 "int8"；GPU 可用 "float16"/"int8_float16"
        """
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_to_text(self, media_path: str, language: Optional[str] = "en") -> str:
        """
        返回纯文本（不含时间戳）。language=None 可自动检测；英文固定用 "en" 更稳更快。
        """
        media_path = os.path.abspath(media_path)
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"Video not found: {media_path}")

        audio_source, cleanup = _extract_wav_if_needed(media_path)
        try:
            segments, _ = self.model.transcribe(
                audio_source,
                language=language,
                vad_filter=True,
                beam_size=5
            )
            lines = [seg.text.strip() for seg in segments if seg.text.strip()]
            return " ".join(lines).strip()
        finally:
            cleanup()

    def transcribe_to_txt_sidecar(self, media_path: str, language: Optional[str] = "en") -> str:
        """
        在视频同目录生成同名 .txt，并返回该 txt 路径。
        """
        text = self.transcribe_to_text(media_path, language=language)
        base, _ = os.path.splitext(os.path.abspath(media_path))
        out_txt = base + ".txt"
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        return out_txt


# 可选：提供一个函数风格的便捷入口
def transcribe_video_to_txt(media_path: str, model_size: str = "small", language: Optional[str] = "en") -> str:
    """
    便捷函数：内部临时加载模型（偶尔用用方便；高频调用建议用 WhisperTranscriber 复用模型）
    """
    wt = WhisperTranscriber(model_size=model_size, device="cpu", compute_type="int8")
    return wt.transcribe_to_txt_sidecar(media_path, language=language)


if __name__ == "__main__":
    # CLI 用法：python transcriber.py <video_path> [model_size]
    if len(sys.argv) < 2:
        print("Usage: python transcriber.py <video_path> [model_size]")
        sys.exit(1)
    video = sys.argv[1]
    size = sys.argv[2] if len(sys.argv) >= 3 else "small"
    out_txt = transcribe_video_to_txt(video, model_size=size, language="en")
    print(f"✅ 转写完成：{out_txt}")
