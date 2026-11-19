from pathlib import Path
import os

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from src.inference import NetraScanInference
from src.explainability import NetraScanXAI
from src.report_generator import generate_report


APP_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = APP_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

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

engine = None
xai_engine = None


def initialize():
    global engine, xai_engine

    if engine is None:
        engine = NetraScanInference()
        xai_engine = NetraScanXAI(engine)

    return engine, xai_engine


def create_probability_chart(predictions):

    fig = go.Figure()

    class_names = [
        "Normal",
        "Immature",
        "Mature"
    ]

    class_keys = [
        "normal",
        "immature",
        "mature"
    ]

    for model_name in MODEL_NAMES:

        probabilities = predictions[
            model_name
        ]["probabilities"]

        values = [
            probabilities[key] * 100
            for key in class_keys
        ]

        fig.add_trace(
            go.Bar(
                name=model_name,
                x=class_names,
                y=values,
                text=[f"{v:.1f}%" for v in values],
                textposition="auto"
            )
        )

    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        title="Class Probability Comparison",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 105],
        xaxis_title="Predicted Class",
        paper_bgcolor="#101625",
        plot_bgcolor="#101625",
        font=dict(color="#E6E9F2"),
        legend=dict(orientation="h", y=1.15),
        margin=dict(l=35, r=20, t=75, b=40),
        height=380
    )

    return fig


def run_analysis(image):

    if image is None:
        raise gr.Error(
            "Please upload or capture an eye image first."
        )

    model_engine, explanation_engine = initialize()

    result = model_engine.predict(image)

    predictions = result["predictions"]

    ensemble_key = result["ensemble_prediction"]

    ensemble_label = CLASS_LABELS.get(
        ensemble_key,
        ensemble_key.title()
    )

    summary = (
        f"## Ensemble Prediction: {ensemble_label}\n\n"
        f"**Model agreement:** {result['agreement']}\n\n"
        "Academic research prototype — not a clinical diagnosis."
    )

    rows = []

    for model_name in MODEL_NAMES:

        details = predictions[model_name]

        rows.append({
            "Model": model_name,
            "Prediction": CLASS_LABELS.get(
                details["predicted_class"],
                details["predicted_class"]
            ),
            "Confidence (%)": round(
                details["confidence"] * 100, 2
            ),
            "Inference (ms)": details["inference_ms"]
        })

    dataframe = pd.DataFrame(rows)

    chart = create_probability_chart(predictions)

    overlay = None
    heatmap = None
    xai_status = "Grad-CAM not generated."

    try:

        explanation = explanation_engine.generate(image)

        overlay = explanation["overlay"]
        heatmap = explanation["heatmap"]

        xai_status = (
            "Swin-Tiny Stage 2 Grad-CAM generated successfully."
        )

    except Exception as error:

        xai_status = (
            f"Grad-CAM unavailable: "
            f"{type(error).__name__}: {error}"
        )

    return (
        image,
        summary,
        dataframe,
        chart,
        overlay,
        heatmap,
        xai_status,
        result
    )


def create_report(result, image):

    if not result:
        raise gr.Error(
            "Run image analysis before generating a report."
        )

    image_name = "uploaded_eye_image.png"

    report = generate_report(
        result=result,
        image_name=image_name,
        save=True
    )

    return report["text"], report["path"]


CUSTOM_CSS = """
.gradio-container {
    max-width: 1350px !important;
    margin: auto !important;
}
#hero {
    text-align: center;
    padding: 18px 0;
}
#hero h1 {
    font-size: 2.5rem;
    font-weight: 800;
    color: #B9A6FF;
}
#hero p {
    color: #BFC7DC;
}
"""

with gr.Blocks(
    title="NetraScan.AI",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate"
    )
) as demo:

    gr.HTML(
        """
        <div id="hero">
            <h1>NetraScan.AI</h1>
            <p>
                Multi-Transformer Cataract Classification
                &amp; Explainable AI
            </p>
        </div>
        """
    )

    prediction_state = gr.State(value=None)

    with gr.Row():

        with gr.Column(scale=1):

            image_input = gr.Image(
                label="Eye Image",
                sources=["upload", "webcam"],
                type="pil",
                height=340
            )

            image_preview = gr.Image(
                label="Selected Image Preview",
                interactive=False,
                height=260
            )

            analyze_button = gr.Button(
                "Analyze Eye Image",
                variant="primary",
                size="lg"
            )

            clear_button = gr.Button(
                "Clear Results",
                variant="secondary"
            )

            stop_button = gr.Button(
                "Stop Server",
                variant="stop"
            )

        with gr.Column(scale=1):

            ensemble_output = gr.Markdown(
                "## Awaiting Image Analysis"
            )

            prediction_table = gr.Dataframe(
                label="Individual Model Predictions",
                interactive=False
            )

    probability_chart = gr.Plot(
        label="Class Probability Distribution"
    )

    gr.Markdown("## Explainable AI — Swin-Tiny Grad-CAM")

    with gr.Row():

        overlay_output = gr.Image(
            label="Grad-CAM Overlay",
            interactive=False,
            height=310
        )

        heatmap_output = gr.Image(
            label="Activation Heatmap",
            interactive=False,
            height=310
        )

    xai_message = gr.Markdown()

    gr.Markdown("## Research Report")

    report_button = gr.Button(
        "Generate Classification Report",
        variant="primary"
    )

    report_preview = gr.Markdown()

    report_download = gr.File(
        label="Download Markdown Report",
        interactive=False
    )

    gr.Markdown(
        """
        ---
        **Research disclaimer:** NetraScan.AI is an
        academic prototype. Predictions and Grad-CAM
        visualizations are not medical diagnoses.
        """
    )

    analyze_button.click(
        fn=run_analysis,
        inputs=[image_input],
        outputs=[
            image_preview,
            ensemble_output,
            prediction_table,
            probability_chart,
            overlay_output,
            heatmap_output,
            xai_message,
            prediction_state
        ]
    )

    report_button.click(
        fn=create_report,
        inputs=[
            prediction_state,
            image_input
        ],
        outputs=[
            report_preview,
            report_download
        ]
    )

    clear_button.click(
        fn=lambda: (
            None, None, "## Awaiting Image Analysis",
            None, None, None, None, "",
            None, "", None
        ),
        inputs=[],
        outputs=[
            image_input,
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
    )


    def stop_server():
        import threading
        import time
        import os

        def shutdown():
            time.sleep(1.5)
            os._exit(0)

        threading.Thread(
            target=shutdown,
            daemon=True
        ).start()

        return "Server shutdown requested."

    stop_button.click(
        fn=stop_server,
        inputs=[],
        outputs=[xai_message]
    )


if __name__ == "__main__":

    print("=" * 65)
    print("NETRASCAN.AI — PORTABLE APPLICATION")
    print("Project:", APP_ROOT)
    print("=" * 65)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
        allowed_paths=[str(REPORTS_DIR)]
    )
