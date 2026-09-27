"""Gradio 本地网页界面：拖入视频 → 得到文字与 JSON。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .ffmpeg_tools import format_hms, format_hms_ms
from .pipeline import (
    ENGINE_FASTER_WHISPER,
    ENGINE_QWEN3_ASR,
    TranscribeOptions,
    is_qwen_engine,
    process,
)
from .transcriber import DEFAULT_MODEL, LANGUAGE_CHOICES, MODEL_CHOICES

#: 界面上的引擎选项（标签 → 引擎标识）。标签里带上取舍，省得用户去翻文档。
ENGINE_LABELS: tuple[tuple[str, str], ...] = (
    ("faster-whisper（默认，无需 torch，CPU 也能跑）", ENGINE_FASTER_WHISPER),
    ("qwen3-asr（中文同音词/符号更准，需独立环境）", ENGINE_QWEN3_ASR),
)

#: 两个引擎各自的默认模型（界面里作为下拉框的默认值）
DEFAULT_MODEL_BY_ENGINE = {
    ENGINE_FASTER_WHISPER: DEFAULT_MODEL,
    ENGINE_QWEN3_ASR: "models/Qwen3-ASR-1.7B",
}


def _engine_value(label_or_id: Optional[str]) -> str:
    """把界面上的引擎标签还原成引擎标识；认不出就退回默认引擎。"""
    if not label_or_id:
        return ENGINE_FASTER_WHISPER
    value = str(label_or_id).strip()
    for label, engine in ENGINE_LABELS:
        if value == label:
            return engine
    return value


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
    default_engine: str = ENGINE_FASTER_WHISPER,
):
    """构建 Gradio 界面对象（同时兼容 Gradio 5.x / 6.x）。"""
    import gradio as gr

    copy_kwargs = _textbox_extra_kwargs(gr)
    # 下拉框的 value 必须是 choices 里的某一项，所以认不出就退回第一项（默认引擎）
    default_engine_id = _engine_value(default_engine)
    default_engine_label = next(
        (label for label, engine in ENGINE_LABELS if engine == default_engine_id),
        ENGINE_LABELS[0][0],
    )

    def run_job(
        file_path: Optional[str],
        engine_label: str,
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

        engine = _engine_value(engine_label)
        task = "translate" if str(task_label).startswith("翻译") else "transcribe"
        if is_qwen_engine(engine) and task == "translate":
            raise gr.Error(
                "Qwen3-ASR 只做原语言转写，不支持翻译。\n"
                "请把「任务」改回转写，或把「识别引擎」换成 faster-whisper。"
            )

        chosen_model = (model or "").strip() or default_model
        # 界面上的模型下拉框列的是 Whisper 家族；切到 Qwen 时若没手动改过，
        # 塞进去的会是 Whisper 的默认名，交给内部按引擎自动选（pipeline 认得这种情况）。
        if is_qwen_engine(engine) and chosen_model == DEFAULT_MODEL:
            chosen_model = ""

        options = TranscribeOptions(
            engine=engine,
            model=chosen_model or DEFAULT_MODEL_BY_ENGINE.get(engine, DEFAULT_MODEL),
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
        status = f"**完成**｜引擎 `{asr.get('engine', '?')}`｜检测语言 `{asr.get('language')}`{confidence}"
        status += (
            f"｜分段 {asr['segments_count']}"
            f"｜音频时长 {format_hms(asr['duration'])}"
            f"｜处理耗时 {asr['elapsed_seconds']:.1f}s"
            f"｜设备 {asr['device']}/{asr['compute_type']}"
            f"｜实时率 {asr['realtime_factor']:.2f}"
        )
        if asr.get("timestamp_source") == "vad-proportional":
            status += "\n\n⚠️ 没有对齐模型，字幕时间是**估算**的（见 docs/troubleshooting.md 7.4）"
        elif asr.get("timestamp_source") == "even-spread":
            status += (
                "\n\n⚠️⚠️ 既没有对齐模型、也没有说话区间，字幕时间是**在整条音轨上平均分配**的，"
                "长静音片段会明显对不上。建议装上 Qwen3-ForcedAligner-0.6B"
                "（见 docs/troubleshooting.md 7.4）"
            )
        status += f"\n\n结果文件：`{result.json_path}`"

        json_path = result.outputs.get("json")
        return status, result.text, rows, (str(json_path) if json_path else None), result.payload

    with gr.Blocks(title="video2context · 视频转文字") as demo:
        gr.Markdown(
            "# video2context · 视频转文字\n"
            "上传视频（中文或英文）→ 自动抽取音频 → 本地识别 → 输出 JSON / 文本 / 字幕。"
            "全程本地运行，视频不出本机。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(
                    label="上传视频或音频",
                    file_types=["video", "audio"],
                    type="filepath",
                )
                engine_in = gr.Dropdown(
                    choices=[label for label, _ in ENGINE_LABELS],
                    value=default_engine_label,
                    label="识别引擎",
                    info="中文课程/讲座（有公式、术语）建议选 qwen3-asr；没显卡或想省事就用默认的",
                )
                model_in = gr.Dropdown(
                    choices=list(MODEL_CHOICES),
                    value=default_model,
                    label="模型（越大越准越慢）",
                    allow_custom_value=True,
                    info="faster-whisper：large-v3 最准、turbo 约 4 倍速；qwen3-asr：填 models/Qwen3-ASR-1.7B（留空则自动选择）",
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
                    "首次运行会自动下载模型（faster-whisper large-v3 约 3GB，"
                    "Qwen3-ASR-1.7B 约 4GB）。\n\n"
                    "国内网络可在命令行用 `--hf-mirror`，或设置环境变量 "
                    "`HF_ENDPOINT=https://hf-mirror.com`。\n\n"
                    "Qwen3-ASR 需要先装独立环境：`scripts\\setup_qwen.ps1`（Windows）"
                    "或 `scripts/setup_qwen.sh`（macOS/Linux）。"
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
            inputs=[
                file_in, engine_in, model_in, lang_in, task_in,
                vad_in, batch_in, device_in, compute_in, keep_in,
            ],
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
    default_engine: str = ENGINE_FASTER_WHISPER,
) -> None:
    """启动网页界面（阻塞）。"""
    _ensure_gradio_temp()
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    demo = build_demo(
        default_model=default_model,
        default_device=default_device,
        output_dir=output_dir,
        default_engine=default_engine,
    )
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=inbrowser,
        show_error=True,
    )
