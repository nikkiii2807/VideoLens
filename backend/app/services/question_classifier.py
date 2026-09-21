import re
from typing import Tuple, Dict
from app.utils.logger import logger

class QuestionClassifier:
    """
    Classifies user question into:
    - VISUAL: appearance, color, object, visibility, shapes
    - ACTION: activities, actions, movements, interactions
    - TEMPORAL: when, timestamps, duration, sequence
    - TRANSCRIPT: spoken words, speaker statements, audio narration
    - OCR: onscreen text, slide titles, signs, written labels
    - MULTIMODAL: reasoning over cause, outcome, full scene description
    """

    PATTERNS = {
        "OCR": [
            r"\b(text|written|sign|slide|words?|title|heading|label|caption|font|letters?)\b",
            r"\bwhat does (it|the (sign|screen|slide|text|label)) say\b",
            r"\bwhat text appears\b",
            r"\bwhat is written\b",
            r"\bread the\b",
        ],
        "TRANSCRIPT": [
            r"\b(say|said|mention(ed)?|speaker|talk(ed|ing)?|discuss(ed)?|state(d)?|explain(ed)?)\b",
            r"\bwhat does the (person|speaker|instructor|narrator) say\b",
            r"\baccording to the (speaker|audio|transcript)\b",
            r"\bwhat was (said|mentioned|spoken)\b",
        ],
        "TEMPORAL": [
            r"\b(when|timestamp|what time|at what second|at what minute|how long|duration|start time|end time)\b",
            r"\bwhen does\b",
            r"\bat what timestamp\b",
            r"\bhow many seconds\b",
        ],
        "ACTION": [
            r"\b(do|does|doing|did|action|perform(s|ed|ing)?|turn(s|ed|ing)?|move(s|ed|ing)?|pick(s|ed|ing)?|lift(s|ed|ing)?|stir(s|ed|ing)?|boil(s|ed|ing)?|drizzle(s|ed|ing)?|avoid(s|ed|ing)?|detect(s|ed|ing)?)\b",
            r"\bwhat does (the person|the rover|the robot|he|she) do\b",
            r"\bwhat happens when\b",
            r"\bdoes (the person|the rover|the robot)\b",
            r"\bhow does (it|the robot|the rover) maneuver\b",
        ],
        "VISUAL": [
            r"\b(color|colour|look like|visible|object|item|see|seen|shape|wearing|shirt|circle|square|triangle|polygon|pot|plate|table|background)\b",
            r"\bwhat color is\b",
            r"\bwhat object is\b",
            r"\bwho is (visible|seen|in the video)\b",
            r"\bwhat is visible\b",
            r"\bwhere is the\b",
            r"\bhow many .* (are visible|can be seen)\b",
        ]
    }

    # Dynamic modality weights by question type (sum to 1.0)
    WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
        "VISUAL": {"visual": 0.55, "text": 0.15, "ocr": 0.20, "temporal": 0.10},
        "ACTION": {"visual": 0.40, "temporal": 0.25, "text": 0.25, "ocr": 0.10},
        "TEMPORAL": {"temporal": 0.35, "visual": 0.30, "text": 0.25, "ocr": 0.10},
        "TRANSCRIPT": {"text": 0.65, "temporal": 0.15, "visual": 0.10, "ocr": 0.10},
        "OCR": {"ocr": 0.60, "visual": 0.20, "text": 0.10, "temporal": 0.10},
        "MULTIMODAL": {"visual": 0.35, "text": 0.35, "ocr": 0.20, "temporal": 0.10},
    }

    @classmethod
    def classify(cls, question: str) -> Tuple[str, Dict[str, float]]:
        q_lower = question.strip().lower()

        # Score pattern hits
        scores: Dict[str, int] = {k: 0 for k in cls.PATTERNS}
        for q_type, patterns in cls.PATTERNS.items():
            for pat in patterns:
                if re.search(pat, q_lower):
                    scores[q_type] += 1

        # Check explicit priority order if patterns matched
        if scores["OCR"] > 0 and any(w in q_lower for w in ["text", "say", "sign", "written", "title", "label"]):
            q_type = "OCR"
        elif scores["TRANSCRIPT"] > 0 and any(w in q_lower for w in ["say", "said", "speaker", "mention", "narrator"]):
            q_type = "TRANSCRIPT"
        elif scores["TEMPORAL"] > 0 and any(w in q_lower for w in ["when", "timestamp", "what time", "at what second"]):
            q_type = "TEMPORAL"
        elif scores["ACTION"] > 0 and any(w in q_lower for w in ["what does", "action", "happen", "move", "turn", "do"]):
            q_type = "ACTION"
        elif scores["VISUAL"] > 0:
            q_type = "VISUAL"
        else:
            max_cat = max(scores, key=scores.get)
            q_type = max_cat if scores[max_cat] > 0 else "MULTIMODAL"

        weights = cls.WEIGHT_PRESETS[q_type]
        logger.info(f"Question classified as '{q_type}' for query: '{question}' (weights: {weights})")
        return q_type, weights

question_classifier = QuestionClassifier()
