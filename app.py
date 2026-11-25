
from pathlib import Path
import atexit
import os
import time
import base64
import mimetypes
import threading
import hashlib
import html

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from src.inference import NetraScanInference
from src.explainability import NetraScanXAI
from src.llm_reporter import LocalLLMReporter
from src.pdf_reporter import generate_pdf_report

APP_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = APP_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

CLASS_LABELS = {
    "normal": "Normal",
    "immature": "Immature Cataract",
    "mature": "Mature Cataract",
    "inconclusive": "Inconclusive"
}

MODEL_NAMES = [
    "ViT-B/16",
    "Swin-Tiny",
    "MobileViT-S"
]

MODEL_COLORS = [
    "#22d3ee",
    "#818cf8",
    "#34d399"
]

engine = None
xai_engine = None
reporter = LocalLLMReporter()
pipeline_lock = threading.RLock()

atexit.register(reporter.close)


def initialize():
    global engine, xai_engine

    if engine is None:
        engine = NetraScanInference()

    if xai_engine is None:
        xai_engine = NetraScanXAI(engine)

    return engine, xai_engine


def image_fingerprint(image):
    image = image.convert("RGB")

    digest = hashlib.sha256()
    digest.update(str(image.size).encode())
    digest.update(image.tobytes())

    return digest.hexdigest()


def create_probability_chart(predictions):
    fig = go.Figure()

    for model, color in zip(MODEL_NAMES, MODEL_COLORS):
        probabilities = predictions[model]["probabilities"]

        fig.add_trace(
            go.Bar(
                name=model,
                x=["Normal", "Immature", "Mature"],
                y=[
                    probabilities["normal"] * 100,
                    probabilities["immature"] * 100,
                    probabilities["mature"] * 100
                ],
                marker_color=color,
                hovertemplate="%{y:.2f}%<extra></extra>"
            )
        )

    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        margin=dict(l=45, r=20, t=65, b=85),
        height=390,
        legend=dict(
            orientation="h",
            y=-0.23,
            x=0.5,
            xanchor="center",
            yanchor="top"
        ),
        yaxis=dict(
            title="Probability (%)",
            range=[0, 105],
            gridcolor="rgba(148,163,184,0.15)"
        )
    )

    return fig


def run_analysis(image):
    if shutdown_requested.is_set():
        raise gr.Error("Application is shutting down.")

    if image is None:
        raise gr.Error("Upload or capture an image first.")

    with pipeline_lock:
        model_engine, explanation_engine = initialize()

        original = image.convert("RGB").copy()
        result = model_engine.predict(original)

        ensemble = CLASS_LABELS[result["ensemble_prediction"]]
        agreement = result["agreement"]

        agreement_message = {
            "3/3": "Unanimous model agreement.",
            "2/3": "Majority prediction — model disagreement present.",
            "1/3": "Inconclusive — no majority agreement."
        }[agreement]

        summary = f"""
### Ensemble classification

# {ensemble}

**Model agreement:** {agreement}

**Agreement status:** {agreement_message}

Research classification only — not a clinical diagnosis.
"""

        rows = []

        for name in MODEL_NAMES:
            prediction = result["predictions"][name]

            rows.append([
                name,
                CLASS_LABELS[prediction["predicted_class"]],
                round(prediction["confidence"] * 100, 2),
                prediction["inference_ms"]
            ])

        table = pd.DataFrame(
            rows,
            columns=[
                "Architecture",
                "Prediction",
                "Confidence (%)",
                "Latency (ms)"
            ]
        )

        chart = create_probability_chart(
            result["predictions"]
        )

        xai_result = None
        overlay = None
        heatmap = None

        try:
            candidate = explanation_engine.generate(original)

            expected_class = result["predictions"][
                "Swin-Tiny"
            ]["predicted_class"]

            expected_index = [
                "normal",
                "immature",
                "mature"
            ].index(expected_class)

            if int(candidate["class_index"]) != expected_index:
                raise ValueError("Grad-CAM target mismatch.")

            xai_result = candidate
            overlay = candidate["overlay"]
            heatmap = candidate["heatmap"]

            xai_status = (
                f"**Grad-CAM target:** "
                f"{CLASS_LABELS[expected_class]}\n\n"
                "Explains the Swin-Tiny prediction, "
                "not the complete ensemble decision."
            )

        except Exception as exc:
            print("Grad-CAM unavailable:", exc)

            xai_status = (
                "Grad-CAM unavailable or target verification "
                "failed. No explainability conclusions presented."
            )

        state = {
            "result": result,
            "original_image": original,
            "fingerprint": image_fingerprint(original),
            "xai_result": xai_result
        }

        return (
            original,
            summary,
            table,
            chart,
            overlay,
            heatmap,
            xai_status,
            state,
            "Analysis complete. Generate the GenAI report below.",
            None
        )


def create_report(state, current_image):
    if shutdown_requested.is_set():
        raise gr.Error("Application is shutting down.")

    if state is None or current_image is None:
        raise gr.Error("Complete image analysis first.")

    if image_fingerprint(current_image) != state["fingerprint"]:
        raise gr.Error("Image changed. Run analysis again.")

    with pipeline_lock:
        result = state["result"]

        genai = reporter.generate(result)

        report = generate_pdf_report(
            result=result,
            original_image=state["original_image"],
            xai_result=state["xai_result"],
            genai=genai,
            image_name="Uploaded or captured image",
            synthetic=False,
            output_dir=REPORTS_DIR
        )

        source = report["report_source"]

        if source == "LLM":
            source_label = (
                "Local GenAI — Qwen2.5-3B-Instruct"
            )
        else:
            source_label = "Deterministic fallback"

        interpretation = html.escape(
            report["interpretation"]
        )

        preview = f"""
### Research interpretation

**Generation source:** {source_label}

{interpretation}

**Report ID:** `{report["report_id"]}`

**Grad-CAM included:** {
    "Yes" if report["xai_available"] else "No"
}

---

Academic research prototype only. Model confidence
does not represent clinical certainty.
"""

        return preview, report["path"]


def reset_analysis():
    return (
        None,
        "### Awaiting image analysis",
        None,
        None,
        None,
        None,
        "Grad-CAM will appear after analysis.",
        None,
        "Generate a report after completing analysis.",
        None
    )


def clear_all():
    return (None,) + reset_analysis()


def shutdown_backend():
    with pipeline_lock:
        reporter.close()

    return "Local GenAI backend stopped. It can restart on demand."



shutdown_requested = threading.Event()


def stop_application():

    if shutdown_requested.is_set():
        return "Application shutdown already requested."

    shutdown_requested.set()

    def shutdown_worker():

        time.sleep(1.5)

        try:
            with pipeline_lock:
                reporter.close()

            demo.close()

        finally:
            os._exit(0)

    threading.Thread(
        target=shutdown_worker,
        daemon=True
    ).start()

    return (
        "Application shutting down. "
        "You may now close this browser tab."
    )


BACKGROUND = None

for filename in [
    "background.png",
    "background.jpg",
    "background.jpeg",
    "background.webp"
]:
    candidate = APP_ROOT / "assets" / filename

    if candidate.is_file():
        mime = mimetypes.guess_type(candidate.name)[0]
        encoded = base64.b64encode(
            candidate.read_bytes()
        ).decode("ascii")

        BACKGROUND = f"data:{mime};base64,{encoded}"
        break


CUSTOM_CSS = """
body, .gradio-container {
    background-color: #050c19 !important;
    color: #e2e8f0 !important;
}

.gradio-container {
    max-width: 1500px !important;
    margin: auto !important;
    min-height: 100vh;
    background-image:
        radial-gradient(
            ellipse at 10% 10%,
            rgba(8,145,178,0.20),
            transparent 45%
        ),
        radial-gradient(
            ellipse at 90% 70%,
            rgba(79,70,229,0.15),
            transparent 45%
        ) !important;
    background-attachment: fixed !important;
}

#hero {
    padding: 30px;
    border: 1px solid rgba(34,211,238,0.30);
    border-radius: 18px;
    background: linear-gradient(
        110deg,
        rgba(7,18,35,0.94),
        rgba(8,47,73,0.78)
    );
    margin-bottom: 16px;
}

#hero h1 {
    font-size: clamp(32px, 5vw, 54px);
    letter-spacing: -1.8px;
    color: #f8fafc;
    margin: 0;
}

#hero h1 span {
    color: #22d3ee;
}

#hero p {
    color: #94a3b8;
    margin: 8px 0;
}

.hero-tag {
    display: inline-block;
    color: #67e8f9;
    border: 1px solid rgba(34,211,238,0.35);
    border-radius: 20px;
    padding: 5px 12px;
    margin: 10px 6px 0 0;
    font-size: 12px;
}

.glass-panel {
    background: rgba(10,22,40,0.76) !important;
    border: 1px solid rgba(34,211,238,0.28) !important;
    border-radius: 16px !important;
    padding: 16px !important;
    backdrop-filter: blur(14px);
}

#ensemble-panel {
    background: rgba(8,47,73,0.40) !important;
    border: 1px solid rgba(34,211,238,0.30) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

#ensemble-panel h1 {
    color: #67e8f9 !important;
}

#analyze-button, #report-button {
    background: linear-gradient(
        110deg,
        #0891b2,
        #2563eb
    ) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    min-height: 46px;
}

#research-footer {
    text-align: center;
    padding: 20px;
    color: #94a3b8;
    font-size: 12px;
}

footer {
    opacity: 0.6;
}
"""

CUSTOM_CSS += '\n#application-controls {\n    align-items: center !important;\n    margin-bottom: 12px !important;\n}\n\n#exit-button {\n    background: linear-gradient(\n        110deg,\n        #991b1b,\n        #dc2626\n    ) !important;\n    color: #ffffff !important;\n    border: 1px solid rgba(248,113,113,0.45) !important;\n    border-radius: 10px !important;\n    font-weight: 700 !important;\n    max-width: 220px !important;\n    min-height: 42px !important;\n}\n\n#exit-button:hover {\n    background: #b91c1c !important;\n    box-shadow: 0 0 18px rgba(239,68,68,0.22);\n}\n\n.glass-panel {\n    box-shadow: 0 8px 30px rgba(0,0,0,0.16);\n}\n\n#hero {\n    box-shadow: 0 8px 35px rgba(8,145,178,0.08);\n}\n\n'

if BACKGROUND:
    CUSTOM_CSS += f"""
    .gradio-container {{
        background-image:
            linear-gradient(
                rgba(5,12,25,0.72),
                rgba(5,12,25,0.85)
            ),
            url("{BACKGROUND}") !important;
        background-size: cover !important;
        background-position: center !important;
    }}
    """


THEME = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="blue",
    neutral_hue="slate"
).set(
    body_background_fill="#050c19",
    body_background_fill_dark="#050c19",
    body_text_color="#e2e8f0",
    body_text_color_dark="#e2e8f0",
    block_background_fill="#0b1628",
    block_background_fill_dark="#0b1628",
    block_label_text_color="#cbd5e1",
    block_label_text_color_dark="#cbd5e1",
    block_title_text_color="#e2e8f0",
    block_title_text_color_dark="#e2e8f0",
    input_background_fill="#101e32",
    input_background_fill_dark="#101e32"
)


with gr.Blocks(
    title="NetraScan.AI V2",
    analytics_enabled=False
) as demo:

    prediction_state = gr.State(None)

    gr.HTML("""
    <div id="hero">
        <h1>NetraScan<span>.AI</span></h1>
        <p>Three Transformers. One Vision.</p>
        <p>
            Multi-transformer cataract classification,
            explainable AI and local generative intelligence.
        </p>
        <span class="hero-tag">VERSION 2.0</span>
        <span class="hero-tag">LOCAL GENAI</span>
        <span class="hero-tag">RESEARCH PROTOTYPE</span>
    </div>
    """)

    with gr.Row(elem_id="application-controls"):

        application_status = gr.Markdown(
            "**Application status:** Online"
        )

        exit_button = gr.Button(
            "Stop Application",
            variant="stop",
            elem_id="exit-button"
        )

    with gr.Row():

        with gr.Column(
            scale=1,
            elem_classes=["glass-panel"]
        ):
            gr.Markdown("## 01 — Image Acquisition")

            image_input = gr.Image(
                sources=["upload", "webcam"],
                type="pil",
                label="Upload or capture an eye image",
                height=300
            )

            analyze_button = gr.Button(
                "Run Transformer Analysis",
                variant="primary",
                elem_id="analyze-button"
            )

            clear_button = gr.Button("Clear Analysis")

            image_preview = gr.Image(
                label="Analyzed input",
                interactive=False,
                height=230
            )

        with gr.Column(
            scale=2,
            elem_classes=["glass-panel"]
        ):
            gr.Markdown("## 02 — Transformer Intelligence")

            ensemble_output = gr.Markdown(
                "### Awaiting image analysis",
                elem_id="ensemble-panel"
            )

            prediction_table = gr.Dataframe(
                headers=[
                    "Architecture",
                    "Prediction",
                    "Confidence (%)",
                    "Latency (ms)"
                ],
                interactive=False
            )

            probability_chart = gr.Plot(
                label="Class probability distribution"
            )

    with gr.Column(elem_classes=["glass-panel"]):

        gr.Markdown("## 03 — Explainable AI")

        with gr.Row():

            overlay_output = gr.Image(
                label="Swin-Tiny Grad-CAM overlay",
                interactive=False,
                height=300
            )

            heatmap_output = gr.Image(
                label="Activation heatmap",
                interactive=False,
                height=300
            )

        xai_message = gr.Markdown(
            "Grad-CAM will appear after analysis."
        )

    with gr.Column(elem_classes=["glass-panel"]):

        gr.Markdown("## 04 — Generative AI Research Report")

        gr.Markdown(
            "**Qwen2.5-3B-Instruct · Q4_K_M · Local inference**"
        )

        report_button = gr.Button(
            "Generate AI Research Report",
            variant="primary",
            elem_id="report-button"
        )

        report_preview = gr.Markdown(
            "Generate a report after completing analysis."
        )

        report_download = gr.File(
            label="Download PDF research report",
            interactive=False
        )

        stop_button = gr.Button(
            "Release Local LLM Resources"
        )

        backend_status = gr.Markdown()

    gr.HTML("""
    <div id="research-footer">
        NetraScan.AI V2 |
        ViT-B/16 + Swin-Tiny + MobileViT-S |
        Academic Research Only<br>
        Not clinically validated for medical diagnosis.
    </div>
    """)

    analysis_outputs = [
        image_preview,
        ensemble_output,
        prediction_table,
        probability_chart,
        overlay_output,
        heatmap_output,
        xai_message,
        prediction_state,
        report_preview,
        report_download
    ]

    analyze_button.click(
        fn=reset_analysis,
        outputs=analysis_outputs,
        queue=False
    ).then(
        fn=run_analysis,
        inputs=[image_input],
        outputs=analysis_outputs,
        concurrency_id="pipeline",
        concurrency_limit=1
    )

    report_button.click(
        fn=create_report,
        inputs=[prediction_state, image_input],
        outputs=[report_preview, report_download],
        concurrency_id="pipeline",
        concurrency_limit=1
    )

    image_input.change(
        fn=reset_analysis,
        outputs=analysis_outputs,
        concurrency_id="pipeline",
        concurrency_limit=1
    )

    clear_button.click(
        fn=clear_all,
        outputs=[image_input] + analysis_outputs,
        concurrency_id="pipeline",
        concurrency_limit=1
    )

    exit_button.click(
        fn=stop_application,
        outputs=[application_status],
        concurrency_id="pipeline",
        concurrency_limit=1
    )

    stop_button.click(
        fn=shutdown_backend,
        outputs=[backend_status],
        concurrency_id="pipeline",
        concurrency_limit=1
    )


if __name__ == "__main__":
    try:
        demo.queue(default_concurrency_limit=1)

        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            inbrowser=True,
            share=False,
            show_error=True,
            allowed_paths=[str(REPORTS_DIR)],
            theme=THEME,
            css=CUSTOM_CSS
        )

    finally:
        reporter.close()
