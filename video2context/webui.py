"""Gradio 本地网页界面：拖入视频 → 得到文字与 JSON。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .ffmpeg_tools import format_hms, format_hms_ms
from .pipeline import TranscribeOptions, process
from .transcriber import DEFAULT_MODEL, LANGUAGE_CHOICES, MODEL_CHOICES


def _ensure_gradio_temp() -> None:
    """把 Gradio 临时目录放到工程内，避免受限系统临时目录带来的问题。"""
    if os.environ.get("GRADIO_TEMP_DIR"):
        return
    try:
        target = Path.cwd() / "output" / ".gradio"
        target.mkdir(parents=True, exist_ok=True)
        os.environ["GRADIO_TEMP_DIR"] = str(target)
    except Exception:
        pass


def _language_value(label_or_code: Optional[str]) -> Optional[str]:
    if not label_or_code:
        return None
    value = str(label_or_code).strip().lower()
    if value in {"auto", "自动检测", "none", ""}:
        return None
    for label, code in LANGUAGE_CHOICES:
        if value == label.strip().lower():
            return code
    return value


def _textbox_extra_kwargs(gr_module) -> dict[str, Any]:
    """Gradio 6 用 ``buttons=["copy"]`` 取代了 5.x 的 ``show_copy_button``。"""
    import inspect

    params = inspect.signature(gr_module.Textbox.__init__).parameters
    if "show_copy_button" in params:
        return {"show_copy_button": True}
    if "buttons" in params:
        return {"buttons": ["copy"]}
    return {}


def build_demo(
    default_model: str = DEFAULT_MODEL,
    default_device: str = "auto",
    output_dir: str = "output",
):
    """构建 Gradio 界面对象（同时兼容 Gradio 5.x / 6.x）。"""
    import gradio as gr

    copy_kwargs = _textbox_extra_kwargs(gr)

    def run_job(
        file_path: Optional[str],
        model: str,
        language: str,
        task_label: str,
        vad: bool,
        batch_size: int,
        device: str,
        compute_type: str,
        keep_audio: bool,
        progress=gr.Progress(),  # noqa: B008 - Gradio 官方要求的写法
    ):
        if not file_path:
            raise gr.Error("请先上传一个视频或音频文件")

        task = "translate" if str(task_label).startswith("翻译") else "transcribe"
        options = TranscribeOptions(
            model=(model or default_model).strip() or default_model,
            device=device or default_device,
            compute_type=(compute_type or "").strip() or None,
            language=_language_value(language),
            task=task,
            vad_filter=bool(vad),
            batch_size=int(batch_size),
            formats=("json", "txt", "srt", "vtt"),
            output_dir=Path(output_dir),
            keep_audio=bool(keep_audio),
        )

        progress(0.03, desc="探测媒体 / 提取音频…")

        def on_progress(processed: float, total: float, _text: str) -> None:
            if total > 0:
                progress(min(0.15 + 0.8 * (processed / total), 0.97),
                         desc=f"识别中 {format_hms(processed)} / {format_hms(total)}")

        try:
            result = process(file_path, options, progress_callback=on_progress)
        except Exception as exc:
            raise gr.Error(f"{type(exc).__name__}: {exc}") from exc

        progress(1.0, desc="完成")
        asr = result.payload["asr"]
        rows = [
            [index, format_hms_ms(seg["start"]), format_hms_ms(seg["end"]), seg["text"]]
            for index, seg in enumerate(result.segments, start=1)
        ]
        probability = asr.get("language_probability")
        confidence = f"（置信度 {probability:.2f}）" if isinstance(probability, (int, float)) else ""
        status = f"**完成**｜检测语言 `{asr.get('language')}`{confidence}"
        status += (
            f"｜分段 {asr['segments_count']}"
            f"｜音频时长 {format_hms(asr['duration'])}"
            f"｜处理耗时 {asr['elapsed_seconds']:.1f}s"
            f"｜设备 {asr['device']}/{asr['compute_type']}"
            f"｜实时率 {asr['realtime_factor']:.2f}\n\n"
            f"结果文件：`{result.json_path}`"
        )

        json_path = result.outputs.get("json")
        return status, result.text, rows, (str(json_path) if json_path else None), result.payload

    with gr.Blocks(title="video2context · 视频转文字") as demo:
        gr.Markdown(
            "# video2context · 视频转文字\n"
            "上传视频（中文或英文）→ 自动抽取音频 → 本地 faster-whisper 识别 → 输出 JSON / 文本 / 字幕。"
            "全程本地运行，视频不出本机。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(
                    label="上传视频或音频",
                    file_types=["video", "audio"],
                    type="filepath",
                )
                model_in = gr.Dropdown(
                    choices=list(MODEL_CHOICES),
                    value=default_model,
                    label="模型（越大越准越慢）",
                    allow_custom_value=True,
                    info="large-v3 精度最高；large-v3-turbo 约 4 倍速；也可填本地模型目录",
                )
                with gr.Row():
                    lang_in = gr.Dropdown(
                        choices=[label for label, _ in LANGUAGE_CHOICES],
                        value="自动检测",
                        label="语言",
                    )
                    device_in = gr.Dropdown(
                        choices=["auto", "cuda", "cpu"],
                        value=default_device,
                        label="设备",
                    )
                task_in = gr.Radio(
                    choices=["转写（保持原语言）", "翻译成英文"],
                    value="转写（保持原语言）",
                    label="任务",
                )
                with gr.Row():
                    vad_in = gr.Checkbox(value=True, label="VAD 静音过滤")
                    keep_in = gr.Checkbox(value=False, label="保留提取出的音频")
                with gr.Row():
                    batch_in = gr.Slider(
                        minimum=-1, maximum=32, step=1, value=-1,
                        label="批量推理大小（-1 自动，0 逐段）",
                    )
                    compute_in = gr.Dropdown(
                        choices=["", "float16", "int8_float16", "int8", "float32"],
                        value="",
                        label="计算精度（留空=自动）",
                    )
                run_btn = gr.Button("开始转写", variant="primary", size="lg")
                gr.Markdown(
                    "首次运行会自动下载模型（large-v3 约 3GB）。"
                    "国内网络可在命令行用 `--hf-mirror`，或设置环境变量 `HF_ENDPOINT=https://hf-mirror.com`。"
                )
            with gr.Column(scale=2):
                status_out = gr.Markdown("等待上传文件…")
                text_out = gr.Textbox(
                    label="转写全文", lines=14, autoscroll=False, **copy_kwargs
                )
                table_out = gr.Dataframe(
                    headers=["#", "开始", "结束", "文本"],
                    datatype=["number", "str", "str", "str"],
                    wrap=True,
                    label="分段结果（含时间戳）",
                )
                with gr.Accordion("结构化 JSON（含时间戳 / 语言 / 参数）", open=False):
                    json_view = gr.JSON(label="JSON")
                file_out = gr.File(label="下载 JSON", interactive=False)

        run_btn.click(
            fn=run_job,
            inputs=[file_in, model_in, lang_in, task_in, vad_in, batch_in, device_in, compute_in, keep_in],
            outputs=[status_out, text_out, table_out, file_out, json_view],
        )
        file_in.change(lambda: "文件已就绪，点击「开始转写」。", outputs=status_out)

    return demo


def launch(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    inbrowser: bool = True,
    default_model: str = DEFAULT_MODEL,
    default_device: str = "auto",
    output_dir: str = "output",
) -> None:
    """启动网页界面（阻塞）。"""
    _ensure_gradio_temp()
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    demo = build_demo(default_model=default_model, default_device=default_device, output_dir=output_dir)
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=inbrowser,
        show_error=True,
    )
