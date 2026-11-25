
from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO
from xml.sax.saxutils import escape
from uuid import uuid4
import math

from PIL import Image as PILImage

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage,
    PageBreak,
    KeepTogether
)

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm

from .llm_reporter import (
    verify_inference_result,
    validate_narrative_v2,
    GROUNDING
)


APP_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = APP_ROOT / "reports"

NAVY = colors.HexColor("#0B1729")
CYAN = colors.HexColor("#0891B2")
BRIGHT_CYAN = colors.HexColor("#22D3EE")
SLATE = colors.HexColor("#475569")
LIGHT = colors.HexColor("#F1F5F9")
BORDER = colors.HexColor("#CBD5E1")
WHITE = colors.white

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

CLASS_NAMES = [
    "normal",
    "immature",
    "mature"
]

CONTENT_WIDTH = 174 * mm


def create_styles():

    return {
        "brand": ParagraphStyle(
            "Brand",
            fontName="Helvetica-Bold",
            fontSize=27,
            leading=32,
            textColor=WHITE
        ),

        "header": ParagraphStyle(
            "Header",
            fontName="Helvetica",
            fontSize=11,
            leading=16,
            textColor=WHITE
        ),

        "heading": ParagraphStyle(
            "Heading",
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=CYAN,
            spaceBefore=12,
            spaceAfter=7
        ),

        "body": ParagraphStyle(
            "Body",
            fontName="Helvetica",
            fontSize=10,
            leading=15,
            textColor=NAVY,
            spaceAfter=6
        ),

        "small": ParagraphStyle(
            "Small",
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=SLATE,
            spaceAfter=4
        ),

        "caption": ParagraphStyle(
            "Caption",
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=SLATE,
            alignment=TA_CENTER
        ),

        "metric_label": ParagraphStyle(
            "MetricLabel",
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=SLATE
        ),

        "metric_value": ParagraphStyle(
            "MetricValue",
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=22,
            textColor=NAVY
        ),

        "table_header": ParagraphStyle(
            "TableHeader",
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=WHITE
        ),

        "table_body": ParagraphStyle(
            "TableBody",
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=NAVY
        )
    }


def paragraph(text, style):

    return Paragraph(
        escape(str(text)),
        style
    )


def image_flowable(image, max_width, max_height):

    if not isinstance(image, PILImage.Image):
        raise TypeError("Expected a PIL image.")

    image = image.convert("RGB").copy()

    image.thumbnail((1200, 1200))

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    width, height = image.size

    scale = min(
        max_width / width,
        max_height / height
    )

    return PDFImage(
        buffer,
        width=width * scale,
        height=height * scale
    )


def styled_table(rows, widths, styles):

    formatted = []

    for index, row in enumerate(rows):

        style = (
            styles["table_header"]
            if index == 0
            else styles["table_body"]
        )

        formatted.append([
            paragraph(value, style)
            for value in row
        ])

    table = Table(
        formatted,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT"
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [WHITE, LIGHT]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("LINEBELOW", (0, -1), (-1, -1), 0.7, CYAN)
        ])
    )

    return table


def resolve_narrative(result, genai):

    agreement = verify_inference_result(result)

    fallback = GROUNDING[agreement]["fallback"]

    if not isinstance(genai, dict):

        return {
            "text": fallback,
            "source": "DETERMINISTIC_FALLBACK",
            "model": "Not used",
            "quantization": "Not applicable",
            "generation_seconds": None
        }

    narrative = genai.get("interpretation", "")

    accepted = (
        genai.get("source") == "LLM"
        and genai.get("llm_accepted") is True
        and genai.get("agreement") == agreement
        and isinstance(narrative, str)
        and all(
            validate_narrative_v2(
                narrative,
                agreement
            ).values()
        )
    )

    if accepted:

        return {
            "text": narrative,
            "source": "LLM",
            "model": str(
                genai.get("model", "Qwen2.5-3B-Instruct")
            ),
            "quantization": str(
                genai.get("quantization", "Q4_K_M")
            ),
            "generation_seconds": genai.get(
                "generation_seconds"
            )
        }

    return {
        "text": fallback,
        "source": "DETERMINISTIC_FALLBACK",
        "model": "Not used",
        "quantization": "Not applicable",
        "generation_seconds": None
    }


def page_decoration(canvas, doc):

    width, height = A4

    canvas.saveState()

    canvas.setFillColor(NAVY)

    canvas.rect(
        0,
        height - 7 * mm,
        width,
        7 * mm,
        fill=1,
        stroke=0
    )

    canvas.setFillColor(CYAN)

    canvas.rect(
        0,
        height - 7 * mm,
        48 * mm,
        7 * mm,
        fill=1,
        stroke=0
    )

    canvas.setStrokeColor(BORDER)

    canvas.line(
        18 * mm,
        16 * mm,
        width - 18 * mm,
        16 * mm
    )

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SLATE)

    canvas.drawString(
        18 * mm,
        11 * mm,
        "NetraScan.AI V2 | Research use only"
    )

    canvas.drawRightString(
        width - 18 * mm,
        11 * mm,
        f"Page {doc.page}"
    )

    canvas.restoreState()


def generate_pdf_report(
    result,
    original_image,
    xai_result=None,
    genai=None,
    image_name="Uploaded image",
    synthetic=False,
    output_dir=None
):

    agreement = verify_inference_result(result)

    if not isinstance(original_image, PILImage.Image):
        raise TypeError("Original image must be a PIL image.")

    narrative = resolve_narrative(
        result,
        genai
    )

    styles = create_styles()

    destination = (
        Path(output_dir)
        if output_dir is not None
        else REPORTS_DIR
    )

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    now = datetime.now(timezone.utc)

    report_id = (
        "NS-"
        + now.strftime("%Y%m%d")
        + "-"
        + uuid4().hex[:8].upper()
    )

    pdf_path = destination / f"{report_id}.pdf"

    temporary_path = destination / f"{report_id}.tmp"

    document = SimpleDocTemplate(
        str(temporary_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=15 * mm,
        bottomMargin=22 * mm,
        title="NetraScan.AI V2 - AI-Assisted Research Report",
        author="NetraScan.AI",
        subject="Multi-transformer cataract classification research"
    )

    story = []

    header_content = [
        Paragraph(
            'NetraScan.<font color="#22D3EE">AI</font>',
            styles["brand"]
        ),

        paragraph(
            "AI-Assisted Research Report",
            styles["header"]
        ),

        paragraph(
            "MULTI-TRANSFORMER CLASSIFICATION | VERSION 2.0",
            styles["header"]
        )
    ]

    header = Table(
        [[header_content]],
        colWidths=[CONTENT_WIDTH]
    )

    header.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), NAVY),
            ("LEFTPADDING", (0, 0), (-1, -1), 15),
            ("RIGHTPADDING", (0, 0), (-1, -1), 15),
            ("TOPPADDING", (0, 0), (-1, -1), 13),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 13)
        ])
    )

    story.append(header)
    story.append(Spacer(1, 8))

    story.append(
        paragraph(
            f"Report ID: {report_id} | "
            f"Generated: {now.strftime('%Y-%m-%d %H:%M UTC')}",
            styles["small"]
        )
    )

    story.append(
        paragraph(
            f"Input: {str(image_name)[:150]}",
            styles["small"]
        )
    )

    if synthetic:

        story.append(
            paragraph(
                "SYNTHETIC SOFTWARE TEST — NOT A CLINICAL CASE",
                styles["heading"]
            )
        )

    story.append(
        paragraph(
            "01 / Classification Summary",
            styles["heading"]
        )
    )

    ensemble = CLASS_LABELS[
        result["ensemble_prediction"]
    ]

    summary = Table(
        [
            [
                paragraph(
                    "Ensemble classification",
                    styles["metric_label"]
                ),
                paragraph(
                    "Model agreement",
                    styles["metric_label"]
                )
            ],
            [
                paragraph(
                    ensemble,
                    styles["metric_value"]
                ),
                paragraph(
                    agreement,
                    styles["metric_value"]
                )
            ]
        ],
        colWidths=[
            112 * mm,
            62 * mm
        ]
    )

    summary.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
            ("LINEBEFORE", (1, 0), (1, -1), 0.5, BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
        ])
    )

    story.append(summary)

    story.append(
        paragraph(
            "02 / Individual Model Predictions",
            styles["heading"]
        )
    )

    prediction_rows = [
        [
            "Architecture",
            "Prediction",
            "Confidence",
            "Latency"
        ]
    ]

    for name in MODEL_NAMES:

        prediction = result["predictions"][name]

        latency = prediction.get("inference_ms")

        if latency is not None:

            latency = float(latency)

            if not math.isfinite(latency) or latency < 0:
                raise ValueError("Invalid inference latency.")

            latency_text = f"{latency:.2f} ms"

        else:

            latency_text = "Not recorded"

        prediction_rows.append([
            name,
            CLASS_LABELS[prediction["predicted_class"]],
            f'{prediction["confidence"] * 100:.2f}%',
            latency_text
        ])

    story.append(
        styled_table(
            prediction_rows,
            [44 * mm, 54 * mm, 37 * mm, 39 * mm],
            styles
        )
    )

    story.append(
        paragraph(
            "03 / Class Probability Distribution",
            styles["heading"]
        )
    )

    probability_rows = [
        [
            "Architecture",
            "Normal",
            "Immature",
            "Mature"
        ]
    ]

    for name in MODEL_NAMES:

        probabilities = result[
            "predictions"
        ][name]["probabilities"]

        probability_rows.append([
            name,
            f'{probabilities["normal"] * 100:.2f}%',
            f'{probabilities["immature"] * 100:.2f}%',
            f'{probabilities["mature"] * 100:.2f}%'
        ])

    story.append(
        styled_table(
            probability_rows,
            [48 * mm, 42 * mm, 42 * mm, 42 * mm],
            styles
        )
    )

    story.append(
        paragraph(
            "04 / Research Interpretation",
            styles["heading"]
        )
    )

    if narrative["source"] == "LLM":

        provenance = (
            f'Local GenAI: {narrative["model"]} | '
            f'{narrative["quantization"]}'
        )

    else:

        provenance = (
            "Deterministic fallback — "
            "no LLM narrative used"
        )

    story.append(
        paragraph(
            provenance,
            styles["small"]
        )
    )

    narrative_box = Table(
        [[
            paragraph(
                narrative["text"],
                styles["body"]
            )
        ]],
        colWidths=[CONTENT_WIDTH]
    )

    narrative_box.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
            ("LINEBEFORE", (0, 0), (0, 0), 3, CYAN),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10)
        ])
    )

    story.append(narrative_box)

    generation_seconds = narrative["generation_seconds"]

    if isinstance(generation_seconds, (int, float)):

        if math.isfinite(generation_seconds) and generation_seconds >= 0:

            story.append(
                paragraph(
                    f"LLM generation latency: "
                    f"{generation_seconds:.3f} seconds",
                    styles["small"]
                )
            )

    story.append(
        paragraph(
            "Research prototype only. Model predictions and "
            "confidence scores do not establish a clinical diagnosis.",
            styles["small"]
        )
    )

    story.append(PageBreak())

    story.append(
        paragraph(
            "05 / Explainable AI",
            styles["heading"]
        )
    )

    story.append(
        paragraph(
            "Swin-Tiny | Gradient-weighted Class Activation Mapping",
            styles["body"]
        )
    )

    xai_available = False
    target_label = None

    if isinstance(xai_result, dict):

        overlay = xai_result.get("overlay")
        heatmap = xai_result.get("heatmap")
        class_index = xai_result.get("class_index")

        if (
            isinstance(overlay, PILImage.Image)
            and isinstance(heatmap, PILImage.Image)
            and isinstance(class_index, int)
            and 0 <= class_index < len(CLASS_NAMES)
        ):

            target_class = CLASS_NAMES[class_index]

            swin_prediction = result[
                "predictions"
            ]["Swin-Tiny"]["predicted_class"]

            if target_class == swin_prediction:

                xai_available = True

                target_label = CLASS_LABELS[
                    target_class
                ]

    if xai_available:

        story.append(
            paragraph(
                f"Grad-CAM target class: {target_label}",
                styles["body"]
            )
        )

        image_table = Table(
            [
                [
                    image_flowable(
                        original_image,
                        51 * mm,
                        51 * mm
                    ),
                    image_flowable(
                        overlay,
                        51 * mm,
                        51 * mm
                    ),
                    image_flowable(
                        heatmap,
                        51 * mm,
                        51 * mm
                    )
                ],
                [
                    paragraph(
                        "Original image",
                        styles["caption"]
                    ),
                    paragraph(
                        "Grad-CAM overlay",
                        styles["caption"]
                    ),
                    paragraph(
                        "Activation heatmap",
                        styles["caption"]
                    )
                ]
            ],
            colWidths=[58 * mm] * 3
        )

        image_table.setStyle(
            TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT)
            ])
        )

        story.append(image_table)

        if synthetic:

            story.append(
                paragraph(
                    "Synthetic test visualizations. These images "
                    "do not represent actual Grad-CAM inference.",
                    styles["small"]
                )
            )

        else:

            story.append(
                paragraph(
                    "The heatmap represents relative activation "
                    "patterns associated with the selected "
                    "Swin-Tiny class. Warmer colors generally "
                    "indicate stronger relative activation.",
                    styles["body"]
                )
            )

        story.append(
            paragraph(
                "Grad-CAM explains the Swin-Tiny prediction, "
                "not the complete ensemble decision. It does "
                "not establish clinically validated anatomical "
                "localization or disease severity.",
                styles["body"]
            )
        )

    else:

        story.append(
            image_flowable(
                original_image,
                75 * mm,
                75 * mm
            )
        )

        story.append(
            paragraph(
                "Grad-CAM is unavailable or its target class "
                "could not be verified. No explainability "
                "conclusions are presented.",
                styles["body"]
            )
        )

    story.append(
        paragraph(
            "06 / Methodology",
            styles["heading"]
        )
    )

    methodology_rows = [
        ["Component", "Method"],
        ["Vision Transformer", "ViT-B/16"],
        ["Hierarchical Transformer", "Swin-Tiny"],
        ["Lightweight Transformer", "MobileViT-S"],
        ["Decision fusion", "Majority voting"],
        ["Explainability", "Swin-Tiny Grad-CAM"],
        ["Narrative generation",
         "Deterministic semantic grounding with local LLM"]
    ]

    story.append(
        styled_table(
            methodology_rows,
            [62 * mm, 112 * mm],
            styles
        )
    )

    story.append(
        paragraph(
            "07 / Research Limitations",
            styles["heading"]
        )
    )

    limitations = [
        "Model confidence is not clinical certainty.",
        "Agreement between models does not guarantee correctness.",
        "Grad-CAM activation maps are not validated pathology maps.",
        "The language model does not independently analyze the image.",
        "Generated narratives undergo heuristic validation; "
        "hallucination resistance is not guaranteed.",
        "NetraScan.AI is not clinically validated for diagnosis."
    ]

    if synthetic:

        limitations.append(
            "This report uses synthetic software-test evidence."
        )

    for item in limitations:

        story.append(
            paragraph(
                "- " + item,
                styles["small"]
            )
        )

    story.append(
        paragraph(
            "This document is intended exclusively for academic "
            "research and software demonstration. It must not be "
            "used to make medical or treatment decisions.",
            styles["body"]
        )
    )

    try:

        document.build(
            story,
            onFirstPage=page_decoration,
            onLaterPages=page_decoration
        )

        pdf_bytes = temporary_path.read_bytes()

        if not pdf_bytes.startswith(b"%PDF-"):
            raise RuntimeError("Invalid PDF header.")

        if not pdf_bytes.rstrip().endswith(b"%%EOF"):
            raise RuntimeError("Invalid PDF EOF marker.")

        temporary_path.replace(pdf_path)

    finally:

        if temporary_path.exists():
            temporary_path.unlink()

    return {
        "path": str(pdf_path),
        "report_id": report_id,
        "ensemble_prediction": ensemble,
        "agreement": agreement,
        "interpretation": narrative["text"],
        "report_source": narrative["source"],
        "xai_available": xai_available,
        "xai_target": target_label,
        "synthetic": bool(synthetic),
        "size_bytes": pdf_path.stat().st_size
    }
