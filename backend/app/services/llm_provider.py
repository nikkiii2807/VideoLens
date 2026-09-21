import os
import re
import json
import base64
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from app.config import settings
from app.utils.logger import logger

SYSTEM_PROMPT = """You are VideoLens, a grounded multimodal video question-answering assistant.

Your job is to answer questions ONLY using the evidence supplied from the video.

The evidence may contain:
- video frames
- timestamps
- transcript segments
- OCR text
- visual descriptions

RULES:

1. Never invent information that is not supported by the supplied evidence.
2. Do not rely on general world knowledge when answering the question.
3. For visual questions, inspect the supplied frames directly.
4. For temporal questions, use the timestamps and nearby evidence.
5. If multiple pieces of evidence disagree, acknowledge the uncertainty.
6. If the supplied evidence is insufficient, explicitly say:
   'I don't have enough evidence from the video to determine that.'
7. Do not assume that something happened merely because it would be likely.
8. Do not confuse similar objects, people, or actions.
9. Only claim an action occurred if the supplied visual or transcript evidence supports it.
10. Keep the answer concise and directly answer the user's question.
11. Always provide the supporting timestamp(s).
12. Never fabricate timestamps.

Your answer must be grounded in the supplied evidence."""

USER_PROMPT_TEMPLATE = """USER QUESTION:
{question}

VIDEO EVIDENCE:

{structured_evidence}

Answer the question using ONLY the evidence above.

If the evidence is insufficient, say that clearly.

Return the answer in this format:

Answer:
<concise grounded answer>

Evidence:
<timestamp(s) and a short explanation of which evidence supports the answer>

Confidence:
<High | Medium | Low>"""


def format_evidence_block(
    evidence: List[Dict[str, Any]],
    temporal_sequence: Optional[Dict[str, Any]] = None
) -> str:
    """Formats retrieved structured evidence into readable text for the prompt."""
    lines = []
    for idx, e in enumerate(evidence[:8], 1):
        t_s = e.get("timestamp_start", e.get("timestamp", 0.0))
        t_e = e.get("timestamp_end", round(t_s + 2.0, 2))
        e_type = e.get("type", "visual")
        content = e.get("text", e.get("content", ""))
        rel = e.get("relevance_score", e.get("score", 0.0))
        
        mins_s = int(t_s // 60)
        secs_s = t_s % 60
        mins_e = int(t_e // 60)
        secs_e = t_e % 60
        ts_str = f"{mins_s:02d}:{secs_s:04.1f} - {mins_e:02d}:{secs_e:04.1f} ({t_s:.1f}s - {t_e:.1f}s)"

        lines.append(f"[{idx}] Type: {e_type} | Timestamp: {ts_str} | Relevance: {rel:.2f}")
        lines.append(f"    Content: {content}")

    if temporal_sequence and temporal_sequence.get("text_progression"):
        lines.append("\nCHRONOLOGICAL EVENT SEQUENCE (TEMPORAL PROGRESSION):")
        lines.append(temporal_sequence["text_progression"])

    return "\n".join(lines)


def parse_grounded_response(raw_text: str, default_timestamps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parses Answer, Evidence, and Confidence from model output."""
    cleaned = raw_text.strip()
    
    # Defaults
    answer = cleaned
    evidence_text = ""
    confidence_level = "Medium"
    confidence_val = 0.70

    # Match Answer:, Evidence:, Confidence:
    ans_match = re.search(r"Answer:\s*(.*?)(?=(?:Evidence:|Confidence:|$))", cleaned, re.DOTALL | re.IGNORECASE)
    ev_match = re.search(r"Evidence:\s*(.*?)(?=(?:Confidence:|$))", cleaned, re.DOTALL | re.IGNORECASE)
    conf_match = re.search(r"Confidence:\s*(High|Medium|Low)", cleaned, re.IGNORECASE)

    if ans_match:
        answer = ans_match.group(1).strip()
    if ev_match:
        evidence_text = ev_match.group(1).strip()
    if conf_match:
        confidence_level = conf_match.group(1).capitalize()
        if confidence_level == "High":
            confidence_val = 0.95
        elif confidence_level == "Medium":
            confidence_val = 0.70
        else:
            confidence_val = 0.30

    # Check for uncertainty phrase
    lower_ans = answer.lower()
    if "don't have enough evidence" in lower_ans or "not enough evidence" in lower_ans or "couldn't find enough reliable evidence" in lower_ans or "cannot determine" in lower_ans:
        confidence_level = "Low"
        confidence_val = 0.25

    # Extract timestamps from evidence and answer
    parsed_timestamps = []
    
    # 1. Look for MM:SS - MM:SS or MM:SS.S
    time_pairs = re.findall(r"(\d{1,2}:\d{2}(?:\.\d+)?)\s*(?:-|to|–)\s*(\d{1,2}:\d{2}(?:\.\d+)?)", evidence_text + " " + answer)
    for s_str, e_str in time_pairs:
        try:
            def to_sec(t_str: str) -> float:
                parts = t_str.split(":")
                return float(parts[0]) * 60 + float(parts[1])
            s_sec = to_sec(s_str)
            e_sec = to_sec(e_str)
            parsed_timestamps.append({
                "start": round(s_sec, 2),
                "end": round(e_sec, 2),
                "description": f"Event interval {s_str} - {e_str}"
            })
        except Exception:
            pass

    # 2. Look for seconds ranges e.g. "0.0s - 4.5s" or "10 to 16 seconds"
    sec_pairs = re.findall(r"(\d+(?:\.\d+)?)\s*s(?:ec)?\s*(?:-|to|–)\s*(\d+(?:\.\d+)?)\s*s(?:ec)?", evidence_text + " " + answer)
    for s_str, e_str in sec_pairs:
        try:
            s_sec = float(s_str)
            e_sec = float(e_str)
            parsed_timestamps.append({
                "start": round(s_sec, 2),
                "end": round(e_sec, 2),
                "description": f"Interval {s_sec:.1f}s - {e_sec:.1f}s"
            })
        except Exception:
            pass

    # 3. If no timestamps extracted but evidence exists, map top evidence items
    if not parsed_timestamps and confidence_level != "Low" and default_timestamps:
        for dt in default_timestamps[:2]:
            t_s = dt.get("timestamp_start", dt.get("timestamp", 0.0))
            t_e = dt.get("timestamp_end", t_s + 2.0)
            parsed_timestamps.append({
                "start": round(float(t_s), 2),
                "end": round(float(t_e), 2),
                "description": f"{dt.get('type', 'video').capitalize()} evidence match"
            })

    return {
        "answer": answer,
        "evidence_summary": evidence_text or "Answer strictly grounded in retrieved video evidence.",
        "timestamps": parsed_timestamps[:3],
        "confidence": confidence_val,
        "confidence_level": confidence_level
    }


class BaseVLMProvider(ABC):
    @abstractmethod
    def generate_grounded_answer(
        self,
        question: str,
        question_type: str,
        evidence: List[Dict[str, Any]],
        episodes: List[Dict[str, float]],
        context_transcripts: List[Dict[str, Any]],
        context_ocr: List[Dict[str, Any]],
        supporting_frames: List[Dict[str, Any]],
        temporal_sequence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generates grounded answer based strictly on retrieved multimodal evidence."""
        pass


class GeminiVLMProvider(BaseVLMProvider):
    """Google Gemini Multimodal Provider using official gemini-2.5-flash with image parts."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = "gemini-2.5-flash"

    def generate_grounded_answer(
        self,
        question: str,
        question_type: str,
        evidence: List[Dict[str, Any]],
        episodes: List[Dict[str, float]],
        context_transcripts: List[Dict[str, Any]],
        context_ocr: List[Dict[str, Any]],
        supporting_frames: List[Dict[str, Any]],
        temporal_sequence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.api_key:
            logger.warning("No GEMINI_API_KEY found, falling back to OfflineHeuristicProvider.")
            return OfflineHeuristicProvider().generate_grounded_answer(
                question, question_type, evidence, episodes, context_transcripts, context_ocr, supporting_frames, temporal_sequence
            )

        # Check evidence sufficiency before invoking API
        if not evidence or (evidence and evidence[0].get("relevance_score", 0.0) < 0.20):
            return {
                "answer": "I couldn't find enough reliable evidence in the video to answer this question.",
                "evidence_summary": "No sufficiently relevant video frames, speech transcripts, or OCR text were retrieved.",
                "timestamps": [],
                "confidence": 0.20,
                "confidence_level": "Low",
                "debug_context": "Evidence below relevance threshold (<0.20)",
                "debug_response": "Uncertainty triggered due to insufficient retrieval evidence."
            }

        # Build structured evidence text
        evidence_text = format_evidence_block(evidence, temporal_sequence)

        # Build full prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question,
            structured_evidence=evidence_text
        )

        full_llm_context = f"=== SYSTEM PROMPT ===\n{SYSTEM_PROMPT}\n\n=== USER PROMPT ===\n{user_prompt}"

        # Construct Gemini API payload with labeled image keyframes
        parts = []

        # Attach keyframes directly as base64 with explicit timestamp annotation
        seen_img_paths = set()
        attached_count = 0
        for f in supporting_frames[:5]:
            img_p = f.get("image_path")
            if not img_p:
                continue
            path_obj = Path(img_p)
            if path_obj.exists() and str(path_obj) not in seen_img_paths:
                seen_img_paths.add(str(path_obj))
                t_val = f.get("timestamp", 0.0)
                try:
                    with open(path_obj, "rb") as img_file:
                        b64 = base64.b64encode(img_file.read()).decode("utf-8")
                    # Label frame timestamp so Gemini grounds the exact second
                    parts.append({
                        "text": f"[Supporting Video Frame at timestamp {t_val:.1f}s (Frame ID: {f.get('frame_id', 'frame')})]"
                    })
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64
                        }
                    })
                    attached_count += 1
                except Exception as e:
                    logger.warning(f"Failed to read image {img_p}: {e}")

        logger.info(f"GeminiVLM: attached {attached_count} visual frames to prompt for '{question}'")

        # Add system instruction & user prompt
        combined_text = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        parts.append({"text": combined_text})

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        # Try calling Gemini with retry on 429 rate limit
        last_error = None
        for attempt in range(2):
            try:
                resp = requests.post(
                    url,
                    json={"contents": [{"parts": parts}]},
                    headers={"Content-Type": "application/json"},
                    timeout=35
                )
                data = resp.json()
                if "error" in data:
                    err_msg = data["error"].get("message", str(data["error"]))
                    if data["error"].get("code") == 429 or "quota exceeded" in err_msg.lower():
                        if attempt == 0:
                            # Parse the suggested retry delay from the error message
                            retry_match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE)
                            wait_sec = float(retry_match.group(1)) if retry_match else 15.0
                            wait_sec = min(wait_sec, 55.0)  # never block more than 55s
                            logger.warning(f"Gemini rate limit hit (429), waiting {wait_sec:.0f}s to retry...")
                            import time as pytime
                            pytime.sleep(wait_sec)
                            continue
                    raise ValueError(f"Gemini API error: {err_msg}")

                if "candidates" not in data or not data["candidates"]:
                    raise ValueError(f"Gemini API returned no candidates: {data}")

                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = parse_grounded_response(raw_text, evidence)
                parsed["debug_context"] = full_llm_context
                parsed["debug_response"] = raw_text
                return parsed

            except Exception as e:
                last_error = e
                if attempt == 1:
                    logger.warning(f"Gemini API invocation failed on attempt {attempt+1}: {e}. Falling back to offline reasoner.")

        fallback = OfflineHeuristicProvider().generate_grounded_answer(
            question, question_type, evidence, episodes, context_transcripts, context_ocr, supporting_frames, temporal_sequence
        )
        fallback["debug_context"] = f"Gemini failed ({str(last_error)}). Context:\n{full_llm_context}"
        return fallback


class OpenAIVLMProvider(BaseVLMProvider):
    """OpenAI GPT-4o Multimodal Provider with base64 image passing."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = "gpt-4o"

    def generate_grounded_answer(
        self,
        question: str,
        question_type: str,
        evidence: List[Dict[str, Any]],
        episodes: List[Dict[str, float]],
        context_transcripts: List[Dict[str, Any]],
        context_ocr: List[Dict[str, Any]],
        supporting_frames: List[Dict[str, Any]],
        temporal_sequence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.api_key:
            return OfflineHeuristicProvider().generate_grounded_answer(
                question, question_type, evidence, episodes, context_transcripts, context_ocr, supporting_frames, temporal_sequence
            )

        evidence_text = format_evidence_block(evidence, temporal_sequence)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question,
            structured_evidence=evidence_text
        )
        full_llm_context = f"=== SYSTEM PROMPT ===\n{SYSTEM_PROMPT}\n\n=== USER PROMPT ===\n{user_prompt}"

        user_contents = []
        for f in supporting_frames[:4]:
            img_p = f.get("image_path")
            if img_p and Path(img_p).exists():
                with open(img_p, "rb") as img_file:
                    b64 = base64.b64encode(img_file.read()).decode("utf-8")
                t_val = f.get("timestamp", 0.0)
                user_contents.append({
                    "type": "text",
                    "text": f"[Video Frame at timestamp {t_val:.1f}s]"
                })
                user_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })

        user_contents.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_contents}
        ]

        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "messages": messages, "temperature": 0.1},
                timeout=30
            )
            raw_text = resp.json()["choices"][0]["message"]["content"]
            parsed = parse_grounded_response(raw_text, evidence)
            parsed["debug_context"] = full_llm_context
            parsed["debug_response"] = raw_text
            return parsed
        except Exception as e:
            logger.warning(f"OpenAI API failed: {e}. Falling back to offline provider.")
            fallback = OfflineHeuristicProvider().generate_grounded_answer(
                question, question_type, evidence, episodes, context_transcripts, context_ocr, supporting_frames, temporal_sequence
            )
            fallback["debug_context"] = f"OpenAI failed ({e}). Context:\n{full_llm_context}"
            return fallback


class OfflineHeuristicProvider(BaseVLMProvider):
    """
    Built-in zero-cost grounded reasoning provider.
    Synthesizes grounded answers strictly from retrieved speech transcripts,
    OCR text detections, and OpenCLIP visual feature scoring.
    Guarantees: Never hallucinates; returns explicit uncertainty when evidence is insufficient.
    """
    def generate_grounded_answer(
        self,
        question: str,
        question_type: str,
        evidence: List[Dict[str, Any]],
        episodes: List[Dict[str, float]],
        context_transcripts: List[Dict[str, Any]],
        context_ocr: List[Dict[str, Any]],
        supporting_frames: List[Dict[str, Any]],
        temporal_sequence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        logger.info(f"Executing OfflineHeuristicProvider grounded reasoning for type '{question_type}'...")

        evidence_text = format_evidence_block(evidence, temporal_sequence)
        full_llm_context = (
            f"=== SYSTEM PROMPT ===\n{SYSTEM_PROMPT}\n\n=== USER PROMPT ===\n"
            f"{USER_PROMPT_TEMPLATE.format(question=question, structured_evidence=evidence_text)}"
        )

        # 1. Check for insufficient evidence
        if not evidence or (evidence and evidence[0].get("relevance_score", 0.0) < 0.22):
            raw_resp = (
                "Answer:\nI don't have enough evidence from the video to determine that.\n\n"
                "Evidence:\nNo sufficiently relevant video frames, speech transcripts, or OCR text were retrieved.\n\n"
                "Confidence:\nLow"
            )
            parsed = parse_grounded_response(raw_resp, [])
            parsed["debug_context"] = full_llm_context
            parsed["debug_response"] = raw_resp
            return parsed

        q_lower = question.lower()
        top_item = evidence[0]
        t_start = top_item.get("timestamp_start", top_item.get("timestamp", 0.0))
        t_end = top_item.get("timestamp_end", t_start + 2.0)
        top_score = top_item.get("relevance_score", 0.5)

        # Detect global / overview queries that should summarize the whole video
        GLOBAL_KEYWORDS = [
            "what happens", "what is happening", "what happened", "summarize", "summary",
            "overview", "all phases", "all steps", "all stages", "describe the video",
            "what does the video show", "what is shown", "tell me about", "what occurs",
            "what take place", "whole video", "entire video", "throughout the video",
            "what are the phases", "what are the stages", "what are the steps"
        ]
        is_global_query = any(kw in q_lower for kw in GLOBAL_KEYWORDS)

        # 2. Check for deliberately unanswerable concepts (e.g. asking about unrelated items not in video)
        all_ev_text = " ".join([e.get("text", "") for e in evidence]).lower()
        stop_words = {
            "what", "when", "does", "where", "which", "about", "there", "video",
            "name", "show", "tell", "with", "from", "that", "this", "have", "been",
            "were", "will", "would", "could", "should", "your", "more", "most", "some",
            "all", "phases", "steps", "stages", "happening", "happens"
        }
        q_words = [w for w in re.findall(r"\w+", q_lower) if len(w) > 2 and w not in stop_words]

        # Stem-aware overlap: 'phases' should match 'phase', 'happening' matches 'happen', etc.
        def words_overlap(q_words: list, ev_text: str) -> bool:
            ev_words = re.findall(r"\w+", ev_text)
            for qw in q_words:
                for ew in ev_words:
                    # exact match or prefix/stem match (handles plurals, -ing, -ed)
                    if qw == ew or ew.startswith(qw) or qw.startswith(ew):
                        return True
            return False

        has_overlap = words_overlap(q_words, all_ev_text) if q_words else True
        max_vis_sim = max([e.get("similarity", 0.0) for e in evidence if e.get("type") == "visual"] or [0.0])

        # Global/overview queries always have "overlap" — we synthesize from all available data
        if is_global_query:
            has_overlap = True

        if not has_overlap and max_vis_sim < 0.32:
            raw_resp = (
                "Answer:\nI don't have enough evidence from the video to determine that.\n\n"
                "Evidence:\nThe requested subject was not found in the video keyframes, transcripts, or OCR detections.\n\n"
                "Confidence:\nLow"
            )
            parsed = parse_grounded_response(raw_resp, [])
            parsed["debug_context"] = full_llm_context
            parsed["debug_response"] = raw_resp
            return parsed

        # 3. Grounded Answer Synthesis by Question Type
        answer_body = ""
        evidence_summary = ""

        if question_type == "TRANSCRIPT" or top_item.get("type") == "transcript":
            # Extract transcript evidence
            tx_items = [e for e in evidence if e.get("type") == "transcript"]
            if tx_items:
                chosen = tx_items[0]
                answer_body = chosen.get("text", "").strip()
                t_start = chosen.get("timestamp_start", t_start)
                t_end = chosen.get("timestamp_end", t_end)
                evidence_summary = f"Spoken transcript at {t_start:.1f}s - {t_end:.1f}s: '{answer_body}'"
            else:
                answer_body = top_item.get("text", "")
                evidence_summary = f"Evidence at {t_start:.1f}s - {t_end:.1f}s: '{answer_body}'"

        elif question_type == "OCR" or top_item.get("type") == "ocr":
            # Extract OCR evidence
            ocr_items = [e for e in evidence if e.get("type") == "ocr"]
            if ocr_items:
                chosen = ocr_items[0]
                answer_body = chosen.get("text", "").strip()
                t_start = chosen.get("timestamp_start", t_start)
                t_end = chosen.get("timestamp_end", t_end)
                evidence_summary = f"Onscreen OCR detected at {t_start:.1f}s: '{answer_body}'"
            else:
                answer_body = top_item.get("text", "")
                evidence_summary = f"Evidence at {t_start:.1f}s: '{answer_body}'"

        elif question_type in ["ACTION", "TEMPORAL"]:
            # Use temporal sequence progression
            if temporal_sequence and temporal_sequence.get("text_progression"):
                prog_lines = [l.strip() for l in temporal_sequence["text_progression"].split("\n") if l.strip()]
                w_start, w_end = temporal_sequence.get("window", (t_start, t_end))
                t_start, t_end = w_start, w_end

                # Find any OCR or speech in this sequence
                active_events = []
                for l in prog_lines:
                    if "OCR:" in l or "Speech:" in l:
                        active_events.append(l)

                if active_events:
                    answer_body = f"Between {t_start:.1f}s and {t_end:.1f}s: {'; '.join(active_events[:3])}"
                else:
                    answer_body = f"The action sequence occurs between {t_start:.1f}s and {t_end:.1f}s."
                evidence_summary = f"Temporal sequence from {t_start:.1f}s to {t_end:.1f}s tracking the event."
            else:
                answer_body = f"The event occurs around {t_start:.1f}s - {t_end:.1f}s: {top_item.get('text', '')}"
                evidence_summary = f"Keyframe evidence at {t_start:.1f}s - {t_end:.1f}s."

        elif question_type == "VISUAL":
            # For visual questions: inspect OCR on the keyframe or visual description
            vis_items = [e for e in evidence if e.get("type") == "visual"]
            chosen = vis_items[0] if vis_items else top_item
            t_start = chosen.get("timestamp_start", t_start)
            t_end = chosen.get("timestamp_end", t_end)
            content = chosen.get("text", "")
            
            # Extract any OCR inside brackets
            ocr_match = re.search(r"\[Onscreen OCR:\s*'([^']+)'\]", content)
            if ocr_match:
                detected_text = ocr_match.group(1)
                answer_body = f"The visual keyframe at {t_start:.1f}s shows {detected_text}."
                evidence_summary = f"Keyframe at {t_start:.1f}s containing visual element '{detected_text}'"
            else:
                answer_body = f"Visual match identified at keyframe {t_start:.1f}s."
                evidence_summary = f"Keyframe visual match at {t_start:.1f}s."

        else:
            # MULTIMODAL — for global/overview queries, synthesize a full chronological summary
            if is_global_query:
                # Build a timeline summary from all available evidence
                all_items = sorted(evidence, key=lambda e: e.get("timestamp_start", e.get("timestamp", 0.0)))

                # Collect all unique OCR texts with timestamps
                ocr_events: list = []
                tx_events: list = []
                seen_ocr: set = set()
                seen_tx: set = set()
                for ev in all_items:
                    ev_text = ev.get("text", "").strip()
                    ts = ev.get("timestamp_start", ev.get("timestamp", 0.0))
                    if ev.get("type") == "ocr" and ev_text and ev_text not in seen_ocr:
                        ocr_events.append((ts, ev_text))
                        seen_ocr.add(ev_text)
                    elif ev.get("type") == "transcript" and ev_text and ev_text not in seen_tx:
                        tx_events.append((ts, ev_text))
                        seen_tx.add(ev_text)
                    # Also parse OCR from visual keyframe content field
                    elif ev.get("type") == "visual":
                        ocr_match = re.search(r"\[Onscreen OCR:\s*'([^']+)'\]", ev_text)
                        if ocr_match:
                            ocr_str = ocr_match.group(1)
                            if ocr_str not in seen_ocr:
                                ocr_events.append((ts, ocr_str))
                                seen_ocr.add(ocr_str)

                # Also pull from context_transcripts and context_ocr passed in
                for tx in context_transcripts:
                    tt = tx.get("start_time", 0.0)
                    txt = tx.get("text", "").strip()
                    if txt and txt not in seen_tx:
                        tx_events.append((tt, txt))
                        seen_tx.add(txt)
                for oc in context_ocr:
                    ot = oc.get("timestamp", 0.0)
                    otxt = oc.get("text", "").strip()
                    if otxt and otxt not in seen_ocr:
                        ocr_events.append((ot, otxt))
                        seen_ocr.add(otxt)

                ocr_events.sort(key=lambda x: x[0])
                tx_events.sort(key=lambda x: x[0])

                # Build answer paragraphs
                summary_parts = []
                if ocr_events:
                    ocr_lines = [f"• {ts:.1f}s: {txt}" for ts, txt in ocr_events[:12]]
                    summary_parts.append("**Onscreen Text (OCR) Timeline:**\n" + "\n".join(ocr_lines))
                if tx_events:
                    tx_lines = [f"• {ts:.1f}s: {txt}" for ts, txt in tx_events[:8]]
                    summary_parts.append("**Speech/Narration Timeline:**\n" + "\n".join(tx_lines))

                if summary_parts:
                    answer_body = "\n\n".join(summary_parts)
                    first_ts = ocr_events[0][0] if ocr_events else (tx_events[0][0] if tx_events else 0.0)
                    last_ts = max(
                        ocr_events[-1][0] if ocr_events else 0.0,
                        tx_events[-1][0] if tx_events else 0.0
                    )
                    t_start, t_end = first_ts, last_ts
                    evidence_summary = f"Full video timeline from {t_start:.1f}s to {t_end:.1f}s synthesized from {len(ocr_events)} OCR events and {len(tx_events)} speech segments."
                else:
                    answer_body = top_item.get("text", "Video content detected but no transcribable text found.")
                    evidence_summary = f"Keyframe evidence at {t_start:.1f}s - {t_end:.1f}s."
            else:
                answer_body = top_item.get("text", "Video content corresponds to the query.")
                evidence_summary = f"Retrieved multimodal evidence at {t_start:.1f}s - {t_end:.1f}s."

        mins_s = int(t_start // 60)
        secs_s = t_start % 60
        mins_e = int(t_end // 60)
        secs_e = t_end % 60
        ts_label = f"{mins_s:02d}:{secs_s:04.1f} - {mins_e:02d}:{secs_e:04.1f}"

        conf_level = "High" if top_score >= 0.50 else "Medium"
        raw_resp = (
            f"Answer:\n{answer_body}\n\n"
            f"Evidence:\n{ts_label}: {evidence_summary}\n\n"
            f"Confidence:\n{conf_level}"
        )

        parsed = parse_grounded_response(raw_resp, evidence)
        parsed["debug_context"] = full_llm_context
        parsed["debug_response"] = raw_resp
        return parsed


def get_llm_provider(provider_name: Optional[str] = None) -> BaseVLMProvider:
    name = (provider_name or settings.LLM_PROVIDER).lower()
    if name == "gemini" and settings.GEMINI_API_KEY:
        return GeminiVLMProvider()
    elif name == "openai" and settings.OPENAI_API_KEY:
        return OpenAIVLMProvider()
    else:
        return OfflineHeuristicProvider()
