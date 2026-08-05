"""
src/api/llm.py

Gemini wrapper (google-genai Python SDK) used at two call sites:

1. Offline/batch (scripts/generate_bio_descriptions.py, and the live-training
   job in src/api/jobs.py) — generate_bio_descriptions_for_rules_dir() makes
   ONE request for every gene in a rules dir's gene_description.csv and ONE
   request for every rule in its rules.json (not one request per item —
   describe_genes_batch/describe_rules_batch), cached to disk
   (gene_description_llm.json, rules_llm.json). No timeout pressure;
   failures are logged and skipped.
2. Live (src/api/inference.py, at predict time) — explain_prediction, wrapped
   in a timeout (default 6s) with a deterministic Vietnamese template fallback so the
   UI never errors or hangs in front of the committee. Grounds its
   conclusion in the CACHED per-rule/per-gene descriptions from (1), not
   just the raw rule text.

Reads GEMINI_API_KEY from the environment (loaded from .env by main.py via
python-dotenv). All functions are safe to call with no key configured — they
raise, and callers are expected to catch/fallback.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# gemini-3.5-flash-lite: an alias-free, non-thinking model — no internal
# "thoughts" tokens eating the output budget (unlike gemini-flash-latest,
# which silently burned ~800/1024 tokens on thinking and truncated replies
# mid-sentence). Faster and cheaper too.
_MODEL = "gemini-3.5-flash-lite"

# Batched requests are chunked at this many items so a single Gemini call
# never has to hold an entire 80-rule dataset's worth of JSON in one
# response (risk of truncation scales with response size).
_BATCH_CHUNK_SIZE = 40


def _client():
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not configured")
    return genai.Client(api_key=api_key)


def _generate(prompt: str, *, max_output_tokens: int = 512) -> str:
    client = _client()
    response = client.models.generate_content(
        model=_MODEL,
        contents=prompt,
        config={"max_output_tokens": max_output_tokens, "temperature": 0.4},
    )
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Empty response from Gemini")
    return text


def _generate_json(prompt: str, *, max_output_tokens: int = 4000) -> Any:
    client = _client()
    response = client.models.generate_content(
        model=_MODEL,
        contents=prompt,
        config={
            "max_output_tokens": max_output_tokens,
            "temperature": 0.4,
            "response_mime_type": "application/json",
        },
    )
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Empty response from Gemini")
    # strict=False: tolerate literal control characters (e.g. an unescaped
    # newline) inside a JSON string value — Gemini's JSON mode doesn't always
    # escape these, and by-the-book strict parsing would otherwise throw the
    # whole chunk away over one cosmetic character.
    return json.loads(text, strict=False)


def _chunks(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


# =============================================================================
# Batched offline descriptions (one request per CHUNK, not per item)
# =============================================================================


def describe_genes_batch(
    gene_rows: List[Dict[str, Any]], log: Callable[[str], None] = lambda msg: None,
) -> Dict[str, str]:
    """``{gene_symbol: description}`` for every row in ``gene_rows`` — one
    Gemini request per chunk of up to _BATCH_CHUNK_SIZE genes, instead of one
    request per gene. A failure in one chunk (bad JSON, network hiccup) is
    logged and skipped — it does NOT discard the other chunks' results, and
    the un-described items just stay missing from the cache for the caller
    to retry next run.
    """
    result: Dict[str, str] = {}
    for chunk in _chunks(gene_rows, _BATCH_CHUNK_SIZE):
        if not chunk:
            continue
        blocks = [
            f"Gene: {row.get('gene', '')}\n"
            f"Tiêu đề: {row.get('gene_title', '')}\n"
            f"GO Biological Process: {row.get('go_biological_process', '')}\n"
            f"GO Cellular Component: {row.get('go_cellular_component', '')}\n"
            f"GO Molecular Function: {row.get('go_molecular_function', '')}\n"
            f"GO Terms (chưa phân loại BP/CC/MF): {row.get('go_terms', '')}\n"
            f"Xuất hiện trong các lớp bệnh: {row.get('classes', '')}"
            for row in chunk
        ]
        prompt = (
            "Bạn là một chuyên gia sinh học phân tử. Với MỖI gene dưới đây, viết MỘT "
            "đoạn văn ngắn (3-4 câu) bằng TIẾNG VIỆT có dấu đầy đủ, mô tả vai trò sinh "
            "học dựa trên dữ liệu được cung cấp riêng cho gene đó. Không bịa đặt thông "
            "tin ngoài dữ liệu; nếu thiếu dữ liệu, chỉ tóm tắt những gì có. Trả lời "
            "dưới dạng JSON object DUY NHẤT, key là chính xác gene symbol đã cho (giữ "
            "nguyên chữ hoa/thường), value là đoạn mô tả (tiếng Việt có dấu).\n\n"
            + "\n\n".join(blocks)
        )
        try:
            data = _generate_json(prompt, max_output_tokens=max(800, 300 * len(chunk)))
            result.update({str(k): v for k, v in data.items()})
        except Exception as e:  # noqa: BLE001
            log(f"[CANH BAO] Bo qua 1 chunk gene ({len(chunk)} gene) do loi: {e}")
    return result


def describe_rules_batch(
    rule_cards: List[Dict[str, Any]], log: Callable[[str], None] = lambda msg: None,
) -> Dict[str, str]:
    """``{str(rule_id): description}`` for every card in ``rule_cards`` — one
    Gemini request per chunk of up to _BATCH_CHUNK_SIZE rules, instead of one
    request per rule. Same per-chunk failure isolation as describe_genes_batch.
    """
    result: Dict[str, str] = {}
    for chunk in _chunks(rule_cards, _BATCH_CHUNK_SIZE):
        if not chunk:
            continue
        blocks = []
        for card in chunk:
            conditions = "; ".join(
                f"{a['gene']} {'cao' if a.get('low') is not None else 'thấp'}"
                for a in card.get("antecedent", [])
            )
            blocks.append(
                f"Rule ID: {card.get('rule_id')}\n"
                f"Luật: {card.get('text', '')}\n"
                f"Các gene liên quan và hướng biểu hiện: {conditions}\n"
                f"Lớp dự đoán: {card.get('consequent', {}).get('class_label', '')}"
            )
        prompt = (
            "Bạn là một chuyên gia tin sinh học (bioinformatics). Với MỖI luật phân "
            "loại dưới đây, viết MỘT đoạn văn ngắn (3-5 câu) bằng TIẾNG VIỆT có dấu "
            "đầy đủ, giải thích Ý NGHĨA SINH HỌC có thể có, dựa trên hướng biểu hiện "
            "các gene và lớp bệnh dự đoán của RIÊNG luật đó. Đây là giả thuyết dựa "
            "trên dữ liệu, không phải kết luận y khoa. Trả lời dưới dạng JSON object "
            "DUY NHẤT, key là Rule ID (dạng chuỗi số) đã cho, value là đoạn giải "
            "thích (tiếng Việt có dấu).\n\n"
            + "\n\n".join(blocks)
        )
        try:
            data = _generate_json(prompt, max_output_tokens=max(1200, 350 * len(chunk)))
            result.update({str(k): v for k, v in data.items()})
        except Exception as e:  # noqa: BLE001
            log(f"[CANH BAO] Bo qua 1 chunk rule ({len(chunk)} rule) do loi: {e}")
    return result


def generate_bio_descriptions_for_rules_dir(
    rules_dir: Path, log: Callable[[str], None] = lambda msg: None,
) -> Dict[str, int]:
    """Generate/update ``gene_description_llm.json`` and ``rules_llm.json``
    next to ``rules_dir``'s ``gene_description.csv``/``rules.json`` — used by
    both the offline script (outputs_holdout/, run once ahead of the defense)
    and the live-training job (outputs_live/<run_id>/, so a "Huan luyen" run
    gets the same descriptions a cached run has). Only sends items not
    already cached, batched (see describe_genes_batch/describe_rules_batch)
    rather than one request per gene/rule. Best-effort: any failure (missing
    API key, network, bad JSON) is logged via ``log`` and never raised.

    Returns ``{"genes": n_new, "rules": n_new}``.
    """
    import pandas as pd

    n_genes = n_rules = 0

    genes_csv = rules_dir / "gene_description.csv"
    if genes_csv.exists():
        out_path = rules_dir / "gene_description_llm.json"
        cache: Dict[str, str] = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
        df = pd.read_csv(genes_csv).fillna("")
        new_rows = [row.to_dict() for _, row in df.iterrows() if row.get("gene") and row["gene"] not in cache]
        if new_rows:
            try:
                new_descriptions = describe_genes_batch(new_rows, log=log)
                cache.update(new_descriptions)
                out_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
                n_genes = len(new_descriptions)
                log(f"[INFO] Gemini: da sinh mo ta cho {n_genes}/{len(new_rows)} gene moi ({len(_chunks(new_rows, _BATCH_CHUNK_SIZE))} request).")
            except Exception as e:  # noqa: BLE001
                log(f"[CANH BAO] Khong the sinh mo ta gene: {e}")

    rules_json = rules_dir / "rules.json"
    if rules_json.exists():
        out_path = rules_dir / "rules_llm.json"
        cache = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
        payload = json.loads(rules_json.read_text(encoding="utf-8"))
        new_rules = [r for r in payload.get("rules", []) if str(r["rule_id"]) not in cache]
        if new_rules:
            try:
                new_descriptions = describe_rules_batch(new_rules, log=log)
                cache.update(new_descriptions)
                out_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
                n_rules = len(new_descriptions)
                log(f"[INFO] Gemini: da sinh mo ta cho {n_rules}/{len(new_rules)} rule moi ({len(_chunks(new_rules, _BATCH_CHUNK_SIZE))} request).")
            except Exception as e:  # noqa: BLE001
                log(f"[CANH BAO] Khong the sinh mo ta rule: {e}")

    return {"genes": n_genes, "rules": n_rules}


# =============================================================================
# Live (predict-time, with fallback)
# =============================================================================


def _fallback_explanation(
    matched_rules: List[Dict[str, Any]],
    classification: str,
    rule_prediction: Optional[str] = None,
    class_votes: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, str]:
    """Deterministic Vietnamese template, same section shape as the Gemini
    path (summary / rationale / model_vs_rule / disclaimer) — always
    available so the UI renders identically whether or not Gemini answered.
    """
    if not matched_rules:
        return {
            "summary": f"Mẫu không khớp bất kỳ luật nào trong tập luật đã trích xuất. Kết quả phân loại của mô hình là {classification}.",
            "rationale": "Không có luật nào khớp nên không có bằng chứng luật để đối chiếu — mô hình quyết định hoàn toàn dựa trên toàn bộ vector đặc trưng đã học, không dựa vào tập luật diễn giải.",
            "model_vs_rule": "",
            "disclaimer": "Đây là một gợi ý hỗ trợ, không phải chẩn đoán y khoa chính thức.",
        }

    genes_mentioned: List[str] = []
    for r in matched_rules:
        for a in r.get("antecedent", []):
            g = a.get("gene")
            if g and g not in genes_mentioned:
                genes_mentioned.append(g)

    summary = (
        f"Dựa trên dữ liệu biểu hiện gene, mô hình học máy phân loại mẫu bệnh phẩm này thuộc "
        f"phân nhóm {classification}, dựa trên {len(matched_rules)} luật khớp trong tập luật đã học."
    )
    rationale = (
        "Ví dụ, các gene như " + ", ".join(genes_mentioned[:3])
        + " có mặt trong một số luật khớp, phản ánh đặc trưng biểu hiện mà mô hình dựa vào để đưa ra dự đoán này."
        if genes_mentioned else ""
    )

    model_vs_rule = _fallback_model_vs_rule(classification, rule_prediction, class_votes)

    return {
        "summary": summary,
        "rationale": rationale,
        "model_vs_rule": model_vs_rule,
        "disclaimer": "Đây là một gợi ý hỗ trợ, không phải chẩn đoán y khoa chính thức.",
    }


def _fallback_model_vs_rule(
    classification: str,
    rule_prediction: Optional[str],
    class_votes: Optional[Dict[str, Dict[str, Any]]],
) -> str:
    if not rule_prediction or not class_votes:
        return ""
    model_pct = class_votes.get(classification, {}).get("percentage")
    rule_pct = class_votes.get(rule_prediction, {}).get("percentage")
    if rule_prediction == classification:
        if len(class_votes) > 1:
            return (
                f"Đa số luật khớp ({rule_pct}%) cũng ủng hộ {classification}, khớp với kết quả của mô hình — "
                "hai nguồn bằng chứng đồng thuận. Tuy vậy vẫn còn một số luật khớp trỏ về lớp khác, phản ánh "
                "việc các phân nhóm có thể chia sẻ một phần đặc trưng sinh học (vd. cùng nhóm thụ thể nội tiết), "
                "nên không nên coi 100% luật đồng thuận là điều kiện bắt buộc để tin tưởng kết quả."
            )
        return (
            f"Toàn bộ luật khớp đều ủng hộ {classification}, khớp hoàn toàn với kết quả của mô hình."
        )
    return (
        f"Mô hình dự đoán {classification}, trong khi đa số luật khớp ({rule_pct}%) lại nghiêng về "
        f"{rule_prediction} (mô hình chỉ có {model_pct or 0}% luật khớp ủng hộ). Sự lệch pha này xảy ra vì "
        "mô hình quyết định dựa trên TOÀN BỘ vector đặc trưng đã học (kể cả các đặc trưng không xuất hiện "
        "trong các luật đơn giản đang hiển thị), còn tỉ lệ luật khớp chỉ phản ánh các điều kiện ngưỡng đơn lẻ "
        "tình cờ đúng với mẫu này — nhiều luật khớp không đồng nghĩa với việc mô hình dùng đúng những luật đó "
        "để quyết định. Đây cũng thường gặp ở các phân nhóm có ranh giới sinh học gần nhau."
    )


async def explain_prediction(
    matched_rules: List[Dict[str, Any]],
    classification: str,
    gene_context: Optional[Dict[str, str]] = None,
    class_votes: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    timeout: float = 6.0,
) -> Dict[str, Any]:
    """Live 'Danh Gia Y Sinh' text for the matched-rule combination of one
    sample — grounded in the CACHED per-rule/per-gene descriptions (from
    generate_bio_descriptions_for_rules_dir) when available, plus the
    matched-rule vote breakdown per class, so the conclusion reflects the
    precomputed biological interpretation instead of re-deriving one from
    scratch off bare rule text.

    Returns structured sections (not one wall-of-text paragraph) so the UI
    can render each with its own heading:
    ``{"summary": str, "rationale": str, "model_vs_rule": str,
    "disclaimer": str, "llm_used": bool}``. Falls back to a deterministic
    template (same shape) on any error or timeout (network, missing key,
    quota, malformed JSON, ...) — never raises, so the predict endpoint
    never fails because of this call.
    """
    import asyncio
    import functools

    gene_context = gene_context or {}
    rule_prediction = next(iter(class_votes), None) if class_votes else None
    rules_desc = "\n".join(
        f"- {r.get('text', '')} (độ tin cậy={r.get('metrics', {}).get('confidence', '?')})"
        + (f"\n  Ý nghĩa sinh học: {r['explanation']}" if r.get("explanation") else "")
        for r in matched_rules
    ) or "(không có luật nào khớp)"
    gene_desc = "\n".join(f"- {g}: {d}" for g, d in gene_context.items()) or "(không có mô tả gene riêng)"
    votes_desc = ""
    if class_votes:
        votes_desc = "\nTỷ lệ luật khớp theo từng lớp (bằng chứng thống kê, độc lập với mô hình):\n" + "\n".join(
            f"- {label}: {v['count']} luật ({v['percentage']}%)" for label, v in class_votes.items()
        )

    agree_note = (
        "Kết quả phân loại của mô hình VÀ nhãn được đa số luật khớp ủng hộ LÀ CÙNG MỘT LỚP."
        if rule_prediction == classification
        else "Kết quả phân loại của mô hình VÀ nhãn được đa số luật khớp ủng hộ LÀ HAI LỚP KHÁC NHAU."
    ) if rule_prediction else ""

    prompt = (
        "Bạn là một chuyên gia bệnh học phân tử hỗ trợ đọc kết quả mô hình học máy phân loại ung thư "
        "từ dữ liệu biểu hiện gene. Trả lời bằng TIẾNG VIỆT có dấu đầy đủ, ở định dạng JSON với đúng 3 "
        "khóa sau (không thêm khóa nào khác, không thêm markdown ```):\n"
        "- \"summary\": 1-2 câu tóm tắt kết quả phân loại của mô hình.\n"
        "- \"rationale\": 3-5 câu luận giải vì sao các luật khớp cho lớp dự đoán của MÔ HÌNH liên quan "
        "đến đặc trưng sinh học của lớp đó — DỰA VÀO các mô tả ý nghĩa sinh học và vai trò gene được "
        "cung cấp dưới đây, không chỉ nhắc lại tên luật. CHỈ chọn ra 2-3 gene/luật TIÊU BIỂU NHẤT làm "
        "ví dụ minh họa (dùng cụm \"ví dụ\", \"chẳng hạn\", \"như\"...) — TUYỆT ĐỐI KHÔNG cố liệt kê hay "
        "tổng hợp ý nghĩa của TẤT CẢ gene/luật khớp, kể cả khi có nhiều luật khớp.\n"
        "- \"model_vs_rule\": 2-4 câu SO SÁNH TƯỜNG MINH giữa kết quả của mô hình và nhãn được đa số "
        "luật khớp ủng hộ. Luôn giải thích RÕ NGUYÊN NHÂN có thể gây nhầm lẫn giữa hai lớp, trong CẢ HAI "
        "trường hợp: (a) nếu hai bên trùng nhau — vẫn nêu vì sao một số luật khớp lại trỏ về lớp khác "
        "(vd. các phân nhóm chia sẻ đặc trưng sinh học chung), và tại sao đồng thuận không có nghĩa là "
        "chắc chắn tuyệt đối; (b) nếu hai bên khác nhau — giải thích tại sao mô hình vẫn có thể đúng dù "
        "đa số luật đơn giản chỉ về lớp khác (mô hình dùng toàn bộ đặc trưng, luật chỉ là điều kiện ngưỡng "
        "đơn lẻ tình cờ khớp), và những gene/đặc trưng nào có thể gây nhầm lẫn giữa hai lớp này.\n"
        "Không thêm khóa \"disclaimer\" — hệ thống tự thêm câu miễn trừ trách nhiệm.\n\n"
        f"Kết quả phân loại của mô hình: {classification}\n"
        + (f"Nhãn được đa số luật khớp ủng hộ: {rule_prediction}\n" if rule_prediction else "")
        + f"{agree_note}\n"
        f"Các luật khớp (kèm ý nghĩa sinh học nếu có):\n{rules_desc}\n"
        f"Vai trò sinh học các gene liên quan:\n{gene_desc}\n"
        f"{votes_desc}\n"
    )

    fallback = _fallback_explanation(matched_rules, classification, rule_prediction, class_votes)

    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(functools.partial(_generate_json, prompt, max_output_tokens=700)),
            timeout=timeout,
        )
        return {
            "summary": data.get("summary") or fallback["summary"],
            "rationale": data.get("rationale") or fallback["rationale"],
            "model_vs_rule": data.get("model_vs_rule") or fallback["model_vs_rule"],
            "disclaimer": fallback["disclaimer"],
            "llm_used": True,
        }
    except Exception:
        return {**fallback, "llm_used": False}
