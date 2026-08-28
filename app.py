import hashlib
import json
import re
import zlib
from datetime import datetime, timezone

from flask import Flask, request, jsonify

app = Flask(__name__)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

URI_RE = re.compile(r"^gs://([^/]+)/(.+)$")
DECIMAL_RE = re.compile(r"^[0-9]+$")

OBJECT_CODES = [
    "URI_INVALID",
    "GENERATION_INVALID",
    "GENERATION_MISMATCH",
    "CRC32C_INVALID",
    "CRC32C_MISMATCH",
    "SCHEMA_INVALID",
    "JSONL_INVALID",
]

ROW_CODES = [
    "DUPLICATE",
    "POLICY_INVALID",
    "OUT_OF_WINDOW",
    "TRAIN_CONTAMINATION",
]


def utf8_key(value):
    return str(value).encode("utf-8")


def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":")
    )


def parse_timestamp(value):
    """
    Accept:
      YYYY-MM-DDTHH:mm:ss[.sss](Z|+HH:mm|-HH:mm)
    and normalize to:
      YYYY-MM-DDTHH:mm:ss.sssZ
    """

    if not isinstance(value, str):
        raise ValueError("invalid timestamp")

    # Require timezone.
    if value.endswith("Z"):
        raw = value[:-1] + "+00:00"
    elif re.search(r"[+-]\d\d:\d\d$", value):
        raw = value
    else:
        raise ValueError("missing timezone")

    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        raise ValueError("invalid timestamp")

    if dt.tzinfo is None:
        raise ValueError("missing timezone")

    dt = dt.astimezone(timezone.utc)

    # Milliseconds exactly 3 digits.
    milliseconds = dt.microsecond // 1000

    return dt.strftime("%Y-%m-%dT%H:%M:%S") + \
        f".{milliseconds:03d}Z"


def parse_timestamp_for_compare(value):
    if not isinstance(value, str):
        raise ValueError("invalid timestamp")

    if value.endswith("Z"):
        raw = value[:-1] + "+00:00"
    else:
        raw = value

    dt = datetime.fromisoformat(raw)

    if dt.tzinfo is None:
        raise ValueError("invalid timestamp")

    return dt.astimezone(timezone.utc)


def valid_generation(value):
    return isinstance(value, str) and DECIMAL_RE.fullmatch(value) is not None


def valid_crc32(value):
    if not isinstance(value, str):
        return False

    # CRC32C is represented as exactly 8 lowercase hex digits.
    return re.fullmatch(r"[0-9a-f]{8}", value) is not None


def crc32c(data):
    """
    The assignment calls the field crc32c.
    Python's zlib implements CRC-32, not CRC-32C.
    This implementation calculates CRC-32C (Castagnoli).
    """

    crc = 0xFFFFFFFF
    polynomial = 0x82F63B78

    for byte in data:
        crc ^= byte

        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ polynomial
            else:
                crc >>= 1

    crc ^= 0xFFFFFFFF

    return f"{crc & 0xFFFFFFFF:08x}"


def tokenize_words(text):
    """
    Lowercase Unicode word-set tokenizer.
    """

    if not isinstance(text, str):
        return set()

    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def jaccard(a, b):
    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def bucket_for_entity(entity):
    digest = hashlib.sha256(entity.encode("utf-8")).digest()

    # First byte interpreted as integer, then modulo 10.
    return digest[0] % 10


def add_reason(target, reason):
    if reason not in target:
        target.append(reason)


# ------------------------------------------------------------
# JSONL processing
# ------------------------------------------------------------

def process_jsonl(content):
    """
    Returns:
      rows
      errors

    A file must contain at least one valid-looking row.
    Blank lines are ignored.
    """

    if not isinstance(content, str):
        return [], ["SCHEMA_INVALID"]

    rows = []
    errors = []

    lines = content.splitlines()

    nonblank = 0

    for line in lines:
        if not line.strip():
            continue

        nonblank += 1

        try:
            obj = json.loads(line)
        except Exception:
            errors.append("JSONL_INVALID")
            continue

        # Each row must be an object.
        if not isinstance(obj, dict):
            errors.append("SCHEMA_INVALID")
            continue

        # Required exact fields.
        required = ["id", "entity", "eventTime", "revision", "text"]

        if any(k not in obj for k in required):
            errors.append("SCHEMA_INVALID")
            continue

        # Fields must have the expected types.
        if not isinstance(obj["id"], str):
            errors.append("SCHEMA_INVALID")
            continue

        if not isinstance(obj["entity"], str):
            errors.append("SCHEMA_INVALID")
            continue

        if not isinstance(obj["eventTime"], str):
            errors.append("SCHEMA_INVALID")
            continue

        if not isinstance(obj["revision"], int) or isinstance(
            obj["revision"], bool
        ):
            errors.append("SCHEMA_INVALID")
            continue

        if obj["revision"] < 0:
            errors.append("SCHEMA_INVALID")
            continue

        if not isinstance(obj["text"], str):
            errors.append("SCHEMA_INVALID")
            continue

        try:
            canonical_time = parse_timestamp(obj["eventTime"])
        except Exception:
            errors.append("SCHEMA_INVALID")
            continue

        row = {
            "id": obj["id"],
            "entity": obj["entity"],
            "eventTime": canonical_time,
            "revision": obj["revision"],
            "text": obj["text"],
        }

        rows.append(row)

    if nonblank == 0:
        errors.append("SCHEMA_INVALID")

    return rows, sorted(set(errors), key=utf8_key)


# ------------------------------------------------------------
# Main endpoint
# ------------------------------------------------------------

@app.post("/build-corpus")
def build_corpus():

    # --------------------------------------------------------
    # Parse request
    # --------------------------------------------------------

    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "INVALID_JSON"}), 400

    if not isinstance(body, dict):
        return jsonify({"error": "INVALID_JSON"}), 400

    # Missing policy or non-array objects -> HTTP 400.
    if "policy" not in body or "objects" not in body:
        return jsonify({"error": "INPUT_INVALID"}), 400

    policy = body["policy"]
    objects = body["objects"]

    if not isinstance(policy, dict) or not isinstance(objects, list):
        return jsonify({"error": "INPUT_INVALID"}), 400

    # --------------------------------------------------------
    # Policy validation
    # --------------------------------------------------------

    policy_invalid = False

    if "minTime" not in policy or "maxTime" not in policy:
        policy_invalid = True
    else:
        try:
            min_time = parse_timestamp_for_compare(policy["minTime"])
            max_time = parse_timestamp_for_compare(policy["maxTime"])

            if min_time > max_time:
                policy_invalid = True
        except Exception:
            policy_invalid = True

    threshold = policy.get("contaminationThreshold")

    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or threshold < 0
        or threshold > 1
    ):
        policy_invalid = True

    # --------------------------------------------------------
    # Object validation
    # --------------------------------------------------------

    accepted_objects = []
    rejected_objects = []

    for obj in objects:

        # URI must be a string.
        uri = obj.get("uri") if isinstance(obj, dict) else None

        reasons = []

        # --------------------------------------------
        # URI
        # --------------------------------------------

        uri_match = (
            isinstance(uri, str)
            and URI_RE.fullmatch(uri) is not None
        )

        if not uri_match:
            reasons.append("URI_INVALID")

        # --------------------------------------------
        # Object shape
        # --------------------------------------------

        if not isinstance(obj, dict):
            rejected_objects.append({
                "uri": None,
                "reasonCodes": ["SCHEMA_INVALID"]
            })
            continue

        # --------------------------------------------
        # generation
        # --------------------------------------------

        generation = obj.get("generation")
        fetched_generation = obj.get("fetchedGeneration")

        generation_valid = valid_generation(generation)
        fetched_generation_valid = valid_generation(fetched_generation)

        if not generation_valid or not fetched_generation_valid:
            reasons.append("GENERATION_INVALID")
        elif generation != fetched_generation:
            reasons.append("GENERATION_MISMATCH")

        # --------------------------------------------
        # schema ID
        # --------------------------------------------

        if obj.get("schemaId") != "training-v1":
            reasons.append("SCHEMA_INVALID")

        # --------------------------------------------
        # CRC
        # --------------------------------------------

        crc_value = obj.get("crc32c")
        content = obj.get("content")

        if not valid_crc32(crc_value):
            reasons.append("CRC32C_INVALID")
        else:
            if not isinstance(content, str):
                reasons.append("SCHEMA_INVALID")
            else:
                calculated = crc32c(content.encode("utf-8"))

                if calculated != crc_value:
                    reasons.append("CRC32C_MISMATCH")

        # --------------------------------------------
        # JSONL
        # --------------------------------------------

        parsed_rows = []

        if not isinstance(content, str):
            if "SCHEMA_INVALID" not in reasons:
                reasons.append("SCHEMA_INVALID")
        else:
            parsed_rows, jsonl_errors = process_jsonl(content)

            for error in jsonl_errors:
                reasons.append(error)

        # --------------------------------------------
        # Object accepted/rejected
        # --------------------------------------------

        reasons = sorted(set(reasons), key=utf8_key)

        if reasons:
            rejected_objects.append({
                "uri": uri if isinstance(uri, str) else None,
                "reasonCodes": reasons
            })
        else:
            accepted_objects.append({
                "uri": uri,
                "generation": generation,
                "fetchedGeneration": fetched_generation,
                "crc32c": crc_value,
                "schemaId": obj["schemaId"],
                "rows": parsed_rows
            })

    # --------------------------------------------------------
    # Flatten accepted rows
    # --------------------------------------------------------

    candidates = []

    for obj in accepted_objects:

        for row in obj["rows"]:

            try:
                event_dt = parse_timestamp_for_compare(
                    row["eventTime"]
                )
            except Exception:
                candidates.append({
                    "row": row,
                    "uri": obj["uri"],
                    "reasonCodes": ["SCHEMA_INVALID"]
                })
                continue

            row_copy = dict(row)
            row_copy["_object_uri"] = obj["uri"]
            row_copy["_event_dt"] = event_dt

            candidates.append({
                "row": row_copy,
                "uri": obj["uri"],
                "reasonCodes": []
            })

    # --------------------------------------------------------
    # Policy / time filtering
    # --------------------------------------------------------

    rejected_rows = []

    valid_candidates = []

    for candidate in candidates:

        row = candidate["row"]

        if policy_invalid:
            rejected_rows.append({
                "id": row.get("id"),
                "reasonCodes": ["POLICY_INVALID"]
            })
            continue

        event_dt = row["_event_dt"]

        if event_dt < min_time or event_dt > max_time:
            rejected_rows.append({
                "id": row.get("id"),
                "reasonCodes": ["OUT_OF_WINDOW"]
            })
            continue

        valid_candidates.append(candidate)

    # --------------------------------------------------------
    # Deduplication
    #
    # JSON tuple:
    # (entity,eventTime,text)
    #
    # Keep highest revision.
    # Tie -> UTF-8-byte-smallest ID.
    # --------------------------------------------------------

    grouped = {}

    for candidate in valid_candidates:

        row = candidate["row"]

        key = (
            row["entity"],
            row["eventTime"],
            row["text"]
        )

        grouped.setdefault(key, []).append(candidate)

    deduplicated = []

    for key, group in grouped.items():

        group.sort(
            key=lambda c: (
                -c["row"]["revision"],
                utf8_key(c["row"]["id"])
            )
        )

        winner = group[0]
        deduplicated.append(winner)

        for loser in group[1:]:

            rejected_rows.append({
                "id": loser["row"]["id"],
                "reasonCodes": ["DUPLICATE"]
            })

    # --------------------------------------------------------
    # Deterministic split
    #
    # 0-5 train
    # 6-7 validation
    # 8-9 test
    # --------------------------------------------------------

    splits = {
        "train": [],
        "validation": [],
        "test": []
    }

    for candidate in deduplicated:

        row = candidate["row"]

        bucket = bucket_for_entity(row["entity"])

        if bucket <= 5:
            split = "train"
        elif bucket <= 7:
            split = "validation"
        else:
            split = "test"

        candidate["split"] = split

        splits[split].append(candidate)

    # --------------------------------------------------------
    # Train contamination
    #
    # Validation/test rows are rejected if their word-set
    # Jaccard similarity with ANY train row is >= threshold.
    # --------------------------------------------------------

    train_candidates = splits["train"]

    train_sets = []

    for candidate in train_candidates:
        train_sets.append(
            tokenize_words(candidate["row"]["text"])
        )

    for split_name in ["validation", "test"]:

        kept = []

        for candidate in splits[split_name]:

            row = candidate["row"]
            row_words = tokenize_words(row["text"])

            contaminated = False

            if not policy_invalid:

                for train_words in train_sets:

                    similarity = jaccard(
                        row_words,
                        train_words
                    )

                    if similarity >= threshold:
                        contaminated = True
                        break

            if contaminated:
                rejected_rows.append({
                    "id": row["id"],
                    "reasonCodes": ["TRAIN_CONTAMINATION"]
                })
            else:
                kept.append(candidate)

        splits[split_name] = kept

    # --------------------------------------------------------
    # Clean rows
    # --------------------------------------------------------

    clean_splits = {
        "train": [],
        "validation": [],
        "test": []
    }

    for split_name, candidates_in_split in splits.items():

        for candidate in candidates_in_split:

            row = candidate["row"]

            clean_splits[split_name].append({
                "id": row["id"],
                "entity": row["entity"],
                "eventTime": row["eventTime"],
                "revision": row["revision"],
                "text": row["text"]
            })

        # Deterministic row order.
        clean_splits[split_name].sort(
            key=lambda r: utf8_key(r["id"])
        )

    # --------------------------------------------------------
    # JSONL bytes + SHA-256
    # --------------------------------------------------------

    digests = {}

    for split_name in ["train", "validation", "test"]:

        lines = []

        for row in clean_splits[split_name]:
            lines.append(compact_json(row))

        if lines:
            jsonl_bytes = (
                "\n".join(lines) + "\n"
            ).encode("utf-8")
        else:
            jsonl_bytes = b""

        digests[split_name] = hashlib.sha256(
            jsonl_bytes
        ).hexdigest()

    # --------------------------------------------------------
    # Lineage
    # --------------------------------------------------------

    lineage = []

    for obj in accepted_objects:

        lineage.append({
            "uri": obj["uri"],
            "generation": obj["generation"],
            "crc32c": obj["crc32c"],
            "schemaId": obj["schemaId"]
        })

    lineage.sort(
        key=lambda x: (
            utf8_key(x["uri"]),
            utf8_key(compact_json(x))
        )
    )

    # --------------------------------------------------------
    # Sort rejected objects
    # --------------------------------------------------------

    rejected_objects.sort(
        key=lambda x: (
            utf8_key(x["uri"] if x["uri"] is not None else ""),
            utf8_key(compact_json(x))
        )
    )

    # --------------------------------------------------------
    # Merge duplicate rejected-row IDs/reasons
    # --------------------------------------------------------

    rejected_map = {}

    for item in rejected_rows:

        row_id = item["id"]

        if row_id not in rejected_map:
            rejected_map[row_id] = set()

        for reason in item["reasonCodes"]:
            rejected_map[row_id].add(reason)

    final_rejected_rows = []

    for row_id, reasons in rejected_map.items():

        final_rejected_rows.append({
            "id": row_id,
            "reasonCodes": sorted(
                reasons,
                key=utf8_key
            )
        })

    final_rejected_rows.sort(
        key=lambda x: (
            utf8_key(
                x["id"] if isinstance(x["id"], str)
                else str(x["id"])
            ),
            utf8_key(compact_json(x))
        )
    )

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    response = {
        "splits": {
            "train": clean_splits["train"],
            "validation": clean_splits["validation"],
            "test": clean_splits["test"]
        },
        "rejectedObjects": rejected_objects,
        "rejectedRows": final_rejected_rows,
        "digests": {
            "train": digests["train"],
            "validation": digests["validation"],
            "test": digests["test"]
        },
        "lineage": lineage
    }

    return app.response_class(
        response=compact_json(response),
        status=200,
        mimetype="application/json"
    )


# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------

@app.get("/")
def home():
    return "JSONL corpus service is running"


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080
    )