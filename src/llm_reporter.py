
from pathlib import Path
from collections import Counter
import subprocess
import requests
import threading
import atexit
import socket
import secrets
import time
import json
import math
import os
import re

APP_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = APP_ROOT / "llm" / "manifest.json"

MODEL_NAMES = (
    "ViT-B/16",
    "Swin-Tiny",
    "MobileViT-S"
)

CLASS_NAMES = (
    "normal",
    "immature",
    "mature"
)


GROUNDING = {'3/3': {'facts': ['All models agree.',
                   'An ensemble classification exists.',
                   'Agreement alone does not establish correctness.'],
         'fallback': 'The models demonstrate unanimous agreement. '
                     'Nevertheless, agreement alone does not establish '
                     'classification correctness.',
         'required_concepts': ['\\b(?:all models|unanimous|unanimity|complete '
                               'agreement)\\b',
                               '\\b(?:correctness|accuracy|correct|accurate|validity)\\b'],
         'forbidden_concepts': ['\\bdisagree',
                                '\\binconclusive\\b',
                                '\\bno majority\\b',
                                '\\bnot unanimous\\b']},
 '2/3': {'facts': ['A majority agreement exists.',
                   'An ensemble classification exists.',
                   'The models are not unanimous.',
                   'The disagreement warrants cautious interpretation.'],
         'fallback': 'A majority agreement establishes an ensemble '
                     'classification, although the models are not unanimous. '
                     'The disagreement warrants cautious interpretation.',
         'required_concepts': ['\\bmajority\\b',
                               '\\b(?:disagreement|disagree|not '
                               'unanimous|dissent)\\b',
                               '\\b(?:caution|cautious|careful|carefully)\\b'],
         'forbidden_concepts': ['\\binconclusive\\b',
                                '\\bno majority\\b',
                                '\\bno ensemble classification\\b',
                                '\\ball models agree\\b',
                                '\\bunanimous agreement\\b',
                                '\\bcannot establish an ensemble\\b']},
 '1/3': {'facts': ['The models disagree.',
                   'No majority agreement exists.',
                   'No ensemble classification can be established.',
                   'The ensemble result is inconclusive.'],
         'fallback': 'The models disagree and no majority agreement exists. '
                     'Consequently, the ensemble result is inconclusive.',
         'required_concepts': ['\\b(?:disagree|disagreement|differ)\\b',
                               '\\b(?:no majority|lack of majority|absence of '
                               'a majority)\\b',
                               '\\binconclusive\\b'],
         'forbidden_concepts': ['\\bmajority supports\\b',
                                '\\bmajority predicts\\b',
                                '\\bmajority agreement exists\\b',
                                '\\bmajority vote establishes\\b',
                                '\\bunanimous agreement\\b']}}

SYSTEM_PROMPT = 'You are an academic scientific writing assistant.\n\nYour only task is to rewrite supplied authoritative facts\ninto a concise professional research paragraph.\n\nThe supplied facts are already verified.\n\nDo not calculate, infer, classify, diagnose or introduce\nnew information.\n\nPreserve the meaning of every supplied fact.\n\nDo not introduce model names, disease names, confidence\nscores, percentages or numerical values.\n\nDo not introduce medical recommendations.\n\nReturn only the requested JSON object.\n\nThe paragraph should contain two or three sentences\nand approximately 25 to 60 words.'

SCHEMA = {'type': 'object',
 'properties': {'interpretation': {'type': 'string'}},
 'required': ['interpretation'],
 'additionalProperties': False}


def validate_narrative_v2(text, agreement):

    normalized = re.sub(
        r"\s+",
        " ",
        text.lower().strip()
    )

    checks = {
        "Non-empty narrative": bool(normalized),

        "Appropriate word count": (
            10 <= len(normalized.split()) <= 85
        ),

        "No numerical values": (
            re.search(r"\d", normalized) is None
        ),

        "No disease or architecture names": (
            re.search(
                r"\b(?:cataract|vit|swin|mobilevit|normal eye)\b",
                normalized
            ) is None
        ),

        "No clinical assertions": (
            re.search(
                r"\b(?:diagnos\w*|patient|surgery|treatment|"
                r"pathology|disease|medication)\b",
                normalized
            ) is None
        )
    }

    if agreement == "3/3":

        checks["Unanimous agreement acknowledged"] = bool(
            re.search(
                r"\b(?:all models|unanimous|unanimity|"
                r"complete agreement)\b",
                normalized
            )
        )

        checks["Correctness limitation acknowledged"] = bool(
            re.search(
                r"\b(?:does not|do not|cannot|"
                r"not necessarily|no guarantee)\b"
                r".{0,100}"
                r"\b(?:correctness|accuracy|correct|accurate|validity)\b",
                normalized
            )
        )

        checks["No contradictory disagreement"] = (
            re.search(
                r"\b(?:disagreement|disagree|inconclusive|"
                r"no majority|not unanimous)\b",
                normalized
            ) is None
        )

    elif agreement == "2/3":

        checks["Majority acknowledged"] = bool(
            re.search(r"\bmajority\b", normalized)
        )

        checks["Disagreement acknowledged"] = bool(
            re.search(
                r"\b(?:disagreement|disagree|"
                r"not unanimous|dissent)\b",
                normalized
            )
        )

        checks["Cautious interpretation acknowledged"] = bool(
            re.search(
                r"\b(?:caution|cautious|careful|carefully)\b",
                normalized
            )
        )

        checks["No contradictory ensemble conclusion"] = (
            re.search(
                r"\b(?:inconclusive|no majority|"
                r"no ensemble classification|"
                r"all models agree|unanimous agreement|"
                r"cannot establish an ensemble)\b",
                normalized
            ) is None
        )

    elif agreement == "1/3":

        checks["Disagreement acknowledged"] = bool(
            re.search(
                r"\b(?:disagree|disagreement|differ)\b",
                normalized
            )
        )

        checks["Absence of majority acknowledged"] = bool(
            re.search(
                r"\b(?:no majority|lack of majority|"
                r"absence of a majority)\b",
                normalized
            )
        )

        checks["Inconclusive result acknowledged"] = bool(
            re.search(
                r"\binconclusive\b",
                normalized
            )
        )

        contradiction_text = re.sub(
            r"\bno majority agreement exists\b",
            "",
            normalized
        )

        checks["No false majority assertion"] = (
            re.search(
                r"\b(?:majority supports|majority predicts|"
                r"majority agreement exists|"
                r"majority vote establishes|"
                r"unanimous agreement|all models agree)\b",
                contradiction_text
            ) is None
        )

    else:

        checks["Valid agreement category"] = False

    checks["No absolute correctness claims"] = (
        re.search(
            r"\b(?:guarantees correctness|"
            r"guarantees accuracy|"
            r"always correct|"
            r"certainly correct|"
            r"proves correctness|"
            r"perfectly accurate)\b",
            normalized
        ) is None
    )

    return checks




def verify_inference_result(result):

    if not isinstance(result, dict):
        raise ValueError("Invalid inference result.")

    predictions = result.get("predictions")

    if not isinstance(predictions, dict):
        raise ValueError("Missing model predictions.")

    if set(predictions) != set(MODEL_NAMES):
        raise ValueError("Unexpected vision model configuration.")

    votes = []

    for name in MODEL_NAMES:

        details = predictions[name]

        prediction = details["predicted_class"]
        confidence = float(details["confidence"])

        if prediction not in CLASS_NAMES:
            raise ValueError(f"Invalid class: {prediction}")

        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"Invalid confidence: {name}")

        probabilities = details["probabilities"]

        if set(probabilities) != set(CLASS_NAMES):
            raise ValueError(f"Invalid probabilities: {name}")

        values = {
            key: float(value)
            for key, value in probabilities.items()
        }

        if not all(
            math.isfinite(value) and 0 <= value <= 1
            for value in values.values()
        ):
            raise ValueError(f"Invalid probability values: {name}")

        if abs(sum(values.values()) - 1.0) > 1e-4:
            raise ValueError(f"Probability normalization failed: {name}")

        if abs(values[prediction] - confidence) > 1e-4:
            raise ValueError(f"Confidence mismatch: {name}")

        if max(values.values()) - confidence > 1e-4:
            raise ValueError(f"Prediction is not an argmax: {name}")

        votes.append(prediction)

    majority_class, majority_count = Counter(
        votes
    ).most_common(1)[0]

    agreement = f"{majority_count}/3"

    ensemble = (
        majority_class
        if majority_count >= 2
        else "inconclusive"
    )

    if result.get("agreement") != agreement:
        raise ValueError("Ensemble agreement mismatch.")

    if result.get("ensemble_prediction") != ensemble:
        raise ValueError("Ensemble prediction mismatch.")

    return agreement


class LocalLLMReporter:

    def __init__(
        self,
        gpu_layers=99,
        context_size=4096,
        threads=8,
        startup_timeout=180,
        generation_timeout=120
    ):

        self.manifest = json.loads(
            MANIFEST_PATH.read_text(encoding="utf-8")
        )

        self.server_exe = (
            APP_ROOT / self.manifest["backend_executable"]
        )

        self.model_path = (
            APP_ROOT / self.manifest["model_path"]
        )

        self.gpu_layers = int(gpu_layers)
        self.context_size = int(context_size)
        self.threads = int(threads)

        self.startup_timeout = startup_timeout
        self.generation_timeout = generation_timeout

        self.process = None
        self.log_file = None
        self.base_url = None
        self.api_key = None

        self.lock = threading.RLock()

        self.session = requests.Session()
        self.session.trust_env = False

        self.log_path = (
            APP_ROOT / "llm" / "runtime" / "server.log"
        )

        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        atexit.register(self.close)


    def _available_port(self):

        with socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        ) as sock:

            sock.bind(("127.0.0.1", 0))

            return sock.getsockname()[1]


    def start(self):

        with self.lock:

            if (
                self.process is not None
                and self.process.poll() is None
                and self.base_url
            ):

                try:

                    response = self.session.get(
                        self.base_url + "/health",
                        timeout=3
                    )

                    if (
                        response.ok
                        and response.json().get("status") == "ok"
                    ):
                        return

                except (requests.RequestException, ValueError):
                    pass

            self.close()

            if not self.server_exe.is_file():
                raise FileNotFoundError(self.server_exe)

            if not self.model_path.is_file():
                raise FileNotFoundError(self.model_path)

            port = self._available_port()

            self.base_url = f"http://127.0.0.1:{port}"
            self.api_key = secrets.token_urlsafe(32)

            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}"
            })

            command = [
                str(self.server_exe),
                "--model", str(self.model_path),
                "--host", "127.0.0.1",
                "--port", str(port),
                "--ctx-size", str(self.context_size),
                "--n-gpu-layers", str(self.gpu_layers),
                "--parallel", "1",
                "--threads", str(self.threads),
                "--api-key", self.api_key
            ]

            environment = os.environ.copy()

            environment["PATH"] = (
                str(self.server_exe.parent)
                + os.pathsep
                + environment.get("PATH", "")
            )

            self.log_file = self.log_path.open(
                "w",
                encoding="utf-8"
            )

            try:

                self.process = subprocess.Popen(
                    command,
                    cwd=str(self.server_exe.parent),
                    stdout=self.log_file,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    creationflags=getattr(
                        subprocess,
                        "CREATE_NO_WINDOW",
                        0
                    )
                )

                started = time.perf_counter()

                while True:

                    if self.process.poll() is not None:

                        raise RuntimeError(
                            "LLM server exited unexpectedly. "
                            f"See {self.log_path}"
                        )

                    try:

                        response = self.session.get(
                            self.base_url + "/health",
                            timeout=2
                        )

                        if (
                            response.ok
                            and response.json().get("status") == "ok"
                        ):
                            return

                    except (requests.RequestException, ValueError):
                        pass

                    if (
                        time.perf_counter() - started
                        > self.startup_timeout
                    ):
                        raise TimeoutError(
                            "LLM initialization timed out."
                        )

                    time.sleep(0.5)

            except BaseException:

                self.close()
                raise


    def close(self):

        with self.lock:

            if self.process is not None:

                if self.process.poll() is None:

                    self.process.terminate()

                    try:

                        self.process.wait(timeout=10)

                    except subprocess.TimeoutExpired:

                        self.process.kill()
                        self.process.wait()

                self.process = None

            if self.log_file is not None:

                self.log_file.close()
                self.log_file = None

            self.base_url = None
            self.api_key = None

            self.session.headers.pop(
                "Authorization",
                None
            )


    def generate(self, result):

        agreement = verify_inference_result(result)

        config = GROUNDING[agreement]

        started = time.perf_counter()

        output = {
            "interpretation": config["fallback"],
            "source": "DETERMINISTIC_FALLBACK",
            "llm_accepted": False,
            "model": self.manifest["model"],
            "quantization": self.manifest["quantization"],
            "agreement": agreement,
            "validation": {},
            "error": None,
            "generation_seconds": None,
            "total_seconds": None,
            "tokens_per_second": None
        }

        with self.lock:

            try:

                self.start()

                facts = "\n".join(
                    "- " + fact
                    for fact in config["facts"]
                )

                payload = {
                    "model": self.manifest["model"],
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": (
                                "Rewrite these verified facts "
                                "without changing their meaning:"
                                "\n\n" + facts
                            )
                        }
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "grounded_narrative",
                            "strict": True,
                            "schema": SCHEMA
                        }
                    },
                    "temperature": 0.0,
                    "seed": 42,
                    "max_tokens": 192,
                    "stream": False
                }

                generation_started = time.perf_counter()

                response = self.session.post(
                    self.base_url + "/v1/chat/completions",
                    json=payload,
                    timeout=self.generation_timeout
                )

                response.raise_for_status()

                output["generation_seconds"] = round(
                    time.perf_counter() - generation_started,
                    3
                )

                raw = response.json()
                choice = raw["choices"][0]

                parsed = json.loads(
                    choice["message"]["content"]
                )

                if (
                    not isinstance(parsed, dict)
                    or set(parsed) != {"interpretation"}
                    or not isinstance(
                        parsed["interpretation"],
                        str
                    )
                ):

                    raise ValueError(
                        "Invalid structured LLM response."
                    )

                narrative = parsed["interpretation"].strip()

                checks = validate_narrative_v2(
                    narrative,
                    agreement
                )

                checks["Generation completed"] = (
                    choice.get("finish_reason") == "stop"
                )

                output["validation"] = checks

                output["tokens_per_second"] = (
                    raw.get("timings", {}).get(
                        "predicted_per_second"
                    )
                )

                if all(checks.values()):

                    output["interpretation"] = narrative
                    output["source"] = "LLM"
                    output["llm_accepted"] = True

                else:

                    output["error"] = (
                        "Generated narrative failed validation."
                    )

            except Exception as error:

                output["error"] = (
                    f"{type(error).__name__}: {error}"
                )

                self.close()

            finally:

                output["total_seconds"] = round(
                    time.perf_counter() - started,
                    3
                )

        return output


    def __enter__(self):

        return self


    def __exit__(self, exc_type, exc_value, traceback):

        self.close()
