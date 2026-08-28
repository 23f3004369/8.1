import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone

from flask import Flask, request

app = Flask(__name__)


# ============================================================
# Constants
# ============================================================

OBJECT_CODES = {
    "URI_INVALID",
    "GENERATION_INVALID",
    "GENERATION_MISMATCH",
    "CRC32C_INVALID",
    "CRC32C_MISMATCH",
    "SCHEMA_INVALID",
    "JSONL_INVALID",
}

ROW_CODES = {
    "DUPLICATE",
    "POLICY_INVALID",
    "OUT_OF_WINDOW",
    "TRAIN_CONTAMINATION",
}


# ============================================================
# Basic deterministic helpers
# ============================================================

def utf8(value):
    return value.encode("utf-8")


def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":")
    )


def sorted_codes(codes):
    return sorted(set(codes), key=utf8)


# ============================================================
# URI validation
# ============================================================

URI_PATTERN = re.compile(
    r"^gs://[^/]+/.+$"
)


def valid_uri(value):
    return isinstance(value, str) and URI_PATTERN.fullmatch(value) is not None


# ============================================================
# Generation validation
# ============================================================

GENERATION_PATTERN = re.compile(r"^[0-9]+$")


def valid_generation(value):
    return (
        isinstance(value, str)
        and GENERATION_PATTERN.fullmatch(value) is not None
    )


# ============================================================
# CRC32C - Castagnoli
# ============================================================

CRC32C_PATTERN = re.compile(r"^[0-9a-f]{8}$")


def valid_crc32c_syntax(value):
    return (
        isinstance(value, str)
        and CRC32C_PATTERN.fullmatch(value) is not None
    )


def crc32c(data):
    """
    CRC-32C / Castagnoli.
    Polynomial in reversed form: 0x82F63B78
    """

    crc = 0xFFFFFFFF

    for byte in data:
        crc ^= byte

        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0x82F63B78
            else:
                crc >>= 1

    crc ^= 0xFFFFFFFF

    return f"{crc:08x}"


# ============================================================
# Timestamp validation and canonicalization
# ============================================================

TIMESTAMP_PATTERN = re.compile(
    r"^"
    r"(\d{4})-(\d{2})-(\d{2})"
    r"T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})"
    r"$"
)


def parse_timestamp(value):
    """
    Strictly accepts:

    YYYY-MM-DDTHH:mm:ssZ
    YYYY-MM-DDTHH:mm:ss.sZ
    YYYY-MM-DDTHH:mm:ss.ssZ
    YYYY-MM-DDTHH:mm:ss.sssZ

    or the same with ±HH:mm.

    Offset magnitude <= 14:00.
    If hour is 14, minutes must be 00.
    """

    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")

    match = TIMESTAMP_PATTERN.fullmatch(value)

    if not match:
        raise ValueError("invalid timestamp syntax")

    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))
    hour = int(match.group(4))
    minute = int(match.group(5))
    second = int(match.group(6))
    fraction = match.group(7)
    offset = match.group(8)

    # Validate calendar/time values.
    if not (0 <= hour <= 23):
        raise ValueError("invalid hour")

    if not (0 <= minute <= 59):
        raise ValueError("invalid minute")

    if not (0 <= second <= 59):
        raise ValueError("invalid second")

    # Validate calendar date.
    try:
        datetime(year, month, day)
    except ValueError:
        raise ValueError("invalid calendar date")

    # Fraction -> exactly milliseconds.
    if fraction is None:
        milliseconds = 0
    elif len(fraction) == 1:
        milliseconds = int(fraction) * 100
    elif len(fraction) == 2:
        milliseconds = int(fraction) * 10
    else:
        milliseconds = int(fraction)

    # Timezone.
    if offset == "Z":
        tz = timezone.utc
    else:
        sign = 1 if offset[0] == "+" else -1

        offset_hour = int(offset[1:3])
        offset_minute = int(offset[4:6])

        if offset_minute > 59:
            raise ValueError("invalid offset minute")

        # Maximum magnitude is 14:00.
        if offset_hour > 14:
            raise ValueError("offset too large")

        if offset_hour == 14 and offset_minute != 0:
            raise ValueError("14 hour offset must have 00 minutes")

        total_minutes = sign * (
            offset_hour * 60 + offset_minute
        )

        from datetime import timedelta

        tz = timezone(timedelta(minutes=total_minutes))

    dt = datetime(
        year,
        month,
        day,
        hour,
        minute,
        second,
        milliseconds * 1000,
        tzinfo=tz
    )

    dt = dt.astimezone(timezone.utc)

    return dt


def canonical_timestamp(value):
    dt = parse_timestamp(value)

    return (
        dt.strftime("%Y-%m-%dT%H:%M:%S")
        + f".{dt.microsecond // 1000:03d}Z"
    )


# ============================================================
# Text canonicalization
# ============================================================

def canonicalize_text(value):
    """
    NFKC
    -> lowercase
    -> trim
    -> collapse Unicode whitespace to ASCII space
    """

    value = unicodedata.normalize("NFKC", value)
    value = value.lower()

    # Unicode whitespace.
    chars = []
    previous_space = False

    for ch in value:
        if ch.isspace():
            if not previous_space:
                chars.append(" ")
            previous_space = True
        else:
            chars.append(ch)
            previous_space = False

    return "".join(chars).strip()


# ============================================================
# Unicode letter/number word-set
# ============================================================

def is_letter_or_number(ch):
    category = unicodedata.category(ch)

    return (
        category.startswith("L")
        or category.startswith("N")
    )


def word_set(text):
    """
    Lowercase Unicode letter/number word-set.

    A word is a maximal consecutive sequence of
    Unicode letters and/or numbers.
    """

    text = text.lower()

    words = []
    current = []

    for ch in text:
        if is_letter_or_number(ch):
            current.append(ch)
        else:
            if current:
                words.append("".join(current))
                current = []

    if current:
        words.append("".join(current))

    return set(words)


def jaccard(a, b):
    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


# ============================================================
# Policy validation
# ============================================================

def validate_policy(policy):
    if not isinstance(policy, dict):
        return False, None, None, None

    if "minTime" not in policy:
        return False, None, None, None

    if "maxTime" not in policy:
        return False, None, None, None

    if "contaminationThreshold" not in policy:
        return False, None, None, None

    try:
        min_dt = parse_timestamp(policy["minTime"])
        max_dt = parse_timestamp(policy["maxTime"])
    except Exception:
        return False, None, None, None

    threshold = policy["contaminationThreshold"]

    # bool is an int subclass, so explicitly reject it.
    if isinstance(threshold, bool):
        return False, None, None, None

    if not isinstance(threshold, (int, float)):
        return False, None, None, None

    if not math.isfinite(float(threshold)):
        return False, None, None, None

    if threshold < 0 or threshold > 1:
        return False, None, None, None

    if min_dt > max_dt:
        return False, None, None, None

    return True, min_dt, max_dt, float(threshold)


# ============================================================
# JSONL row parsing
# ============================================================

EXPECTED_ROW_KEYS = {
    "id",
    "entity",
    "eventTime",
    "revision",
    "text",
}


def valid_safe_revision(value):
    """
    Non-negative safe integer.
    JS safe integer maximum = 2^53 - 1.
    """

    if isinstance(value, bool):
        return False

    if not isinstance(value, int):
        return False

    return 0 <= value <= (2**53 - 1)


def parse_jsonl(content):
    """
    Returns:
        rows,
        has_json_error,
        has_schema_error,
        has_any_nonblank_line
    """

    rows = []
    has_json_error = False
    has_schema_error = False
    has_any_nonblank_line = False

    # splitlines() handles normal newline variants.
    for line in content.splitlines():

        if line.strip() == "":
            continue

        has_any_nonblank_line = True

        try:
            parsed = json.loads(line)
        except Exception:
            has_json_error = True
            continue

        # Every parsed line must be an object.
        if not isinstance(parsed, dict):
            has_schema_error = True
            continue

        # EXACTLY these five keys.
        if set(parsed.keys()) != EXPECTED_ROW_KEYS:
            has_schema_error = True
            continue

        # Four text fields must be strings.
        if not isinstance(parsed["id"], str):
            has_schema_error = True
            continue

        if not isinstance(parsed["entity"], str):
            has_schema_error = True
            continue

        if not isinstance(parsed["eventTime"], str):
            has_schema_error = True
            continue

        if not isinstance(parsed["text"], str):
            has_schema_error = True
            continue

        # Revision must be non-negative safe integer.
        if not valid_safe_revision(parsed["revision"]):
            has_schema_error = True
            continue

        # eventTime must be valid.
        try:
            event_time = canonical_timestamp(parsed["eventTime"])
        except Exception:
            has_schema_error = True
            continue

        rows.append({
            "id": parsed["id"],
            "entity": parsed["entity"],
            "eventTime": event_time,
            "revision": parsed["revision"],
            "text": parsed["text"],
        })

    # Empty/blank-only file.
    if not has_any_nonblank_line:
        has_schema_error = True

    return (
        rows,
        has_json_error,
        has_schema_error,
        has_any_nonblank_line
    )


# ============================================================
# Object processing
# ============================================================

def process_object(obj):
    """
    Returns:

        accepted_object or None
        rejected_object
    """

    # Non-object supplied.
    if not isinstance(obj, dict):
        return (
            None,
            {
                "uri": None,
                "reasonCodes": ["SCHEMA_INVALID"]
            }
        )

    reasons = []

    # --------------------------------------------------------
    # URI
    # --------------------------------------------------------

    uri = obj.get("uri")

    if not valid_uri(uri):
        reasons.append("URI_INVALID")

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    generation = obj.get("generation")
    fetched_generation = obj.get("fetchedGeneration")

    generation_valid = valid_generation(generation)
    fetched_generation_valid = valid_generation(fetched_generation)

    if not generation_valid or not fetched_generation_valid:
        reasons.append("GENERATION_INVALID")

    # Unequal supplied generation values.
    if (
        "generation" in obj
        and "fetchedGeneration" in obj
        and generation != fetched_generation
    ):
        reasons.append("GENERATION_MISMATCH")

    # --------------------------------------------------------
    # CRC32C
    # --------------------------------------------------------

    crc_value = obj.get("crc32c")

    crc_valid = valid_crc32c_syntax(crc_value)

    if not crc_valid:
        reasons.append("CRC32C_INVALID")

    # --------------------------------------------------------
    # Schema ID
    # --------------------------------------------------------

    if obj.get("schemaId") != "training-v1":
        reasons.append("SCHEMA_INVALID")

    # --------------------------------------------------------
    # Content
    # --------------------------------------------------------

    content = obj.get("content")

    if not isinstance(content, str):
        reasons.append("SCHEMA_INVALID")

        # IMPORTANT:
        # CRC32C_MISMATCH is NOT checked unless content is
        # a string and CRC syntax is valid.
    elif crc_valid:
        calculated_crc = crc32c(content.encode("utf-8"))

        if calculated_crc != crc_value:
            reasons.append("CRC32C_MISMATCH")

    # --------------------------------------------------------
    # JSONL
    # --------------------------------------------------------

    rows = []

    if isinstance(content, str):
        (
            rows,
            has_json_error,
            has_schema_error,
            _
        ) = parse_jsonl(content)

        if has_json_error:
            reasons.append("JSONL_INVALID")

        if has_schema_error:
            reasons.append("SCHEMA_INVALID")

    # --------------------------------------------------------
    # Accepted/rejected
    # --------------------------------------------------------

    reasons = sorted_codes(reasons)

    if reasons:
        return (
            None,
            {
                "uri": uri if isinstance(uri, str) else None,
                "reasonCodes": reasons
            }
        )

    return (
        {
            "uri": uri,
            "generation": generation,
            "crc32c": crc_value,
            "schemaId": "training-v1",
            "rows": rows,
        },
        None
    )


# ============================================================
# Deterministic bucket
# ============================================================

def split_for_entity(entity):
    digest = hashlib.sha256(
        entity.encode("utf-8")
    ).digest()

    bucket = digest[0] % 10

    if bucket <= 5:
        return "train"

    if bucket <= 7:
        return "validation"

    return "test"


# ============================================================
# Main endpoint
# ============================================================

@app.route("/build-corpus", methods=["POST"])
def build_corpus():

    # --------------------------------------------------------
    # Request parsing
    # --------------------------------------------------------

    if not request.is_json:
        return app.response_class(
            response='{"error":"INVALID_INPUT"}',
            status=400,
            mimetype="application/json"
        )

    try:
        body = request.get_json()
    except Exception:
        return app.response_class(
            response='{"error":"INVALID_INPUT"}',
            status=400,
            mimetype="application/json"
        )

    if not isinstance(body, dict):
        return app.response_class(
            response='{"error":"INVALID_INPUT"}',
            status=400,
            mimetype="application/json"
        )

    # Missing policy or non-array objects -> exactly 400.
    if "policy" not in body:
        return app.response_class(
            response='{"error":"INVALID_INPUT"}',
            status=400,
            mimetype="application/json"
        )

    if "objects" not in body:
        return app.response_class(
            response='{"error":"INVALID_INPUT"}',
            status=400,
            mimetype="application/json"
        )

    if not isinstance(body["objects"], list):
        return app.response_class(
            response='{"error":"INVALID_INPUT"}',
            status=400,
            mimetype="application/json"
        )

    policy = body["policy"]
    objects = body["objects"]

    # --------------------------------------------------------
    # Policy
    # --------------------------------------------------------

    (
        policy_valid,
        min_time,
        max_time,
        threshold
    ) = validate_policy(policy)

    # --------------------------------------------------------
    # Validate objects
    # --------------------------------------------------------

    accepted_objects = []
    rejected_objects = []

    for obj in objects:

        accepted, rejected = process_object(obj)

        if rejected is not None:
            rejected_objects.append(rejected)
        else:
            accepted_objects.append(accepted)

    # --------------------------------------------------------
    # Collect rows from accepted objects
    # --------------------------------------------------------

    candidates = []

    for obj in accepted_objects:

        for row in obj["rows"]:

            # Canonicalize entity/text before deduplication.
            entity = canonicalize_text(row["entity"])
            text = canonicalize_text(row["text"])

            canonical_row = {
                "id": row["id"],
                "entity": entity,
                "eventTime": row["eventTime"],
                "revision": row["revision"],
                "text": text,
            }

            candidates.append(canonical_row)

    # --------------------------------------------------------
    # Deduplication
    #
    # Key:
    # [entity,eventTime,text]
    #
    # Highest revision wins.
    # Tie -> UTF-8-smallest ID.
    # --------------------------------------------------------

    groups = {}

    for row in candidates:

        key = (
            row["entity"],
            row["eventTime"],
            row["text"]
        )

        groups.setdefault(key, []).append(row)

    retained = []
    rejected_rows = []

    for group in groups.values():

        group.sort(
            key=lambda row: (
                -row["revision"],
                utf8(row["id"])
            )
        )

        winner = group[0]
        retained.append(winner)

        for loser in group[1:]:
            rejected_rows.append({
                "id": loser["id"],
                "reasonCodes": ["DUPLICATE"]
            })

    # --------------------------------------------------------
    # Policy / time filtering
    # --------------------------------------------------------

    policy_retained = []

    for row in retained:

        if not policy_valid:

            rejected_rows.append({
                "id": row["id"],
                "reasonCodes": ["POLICY_INVALID"]
            })

            continue

        event_dt = parse_timestamp(row["eventTime"])

        if event_dt < min_time or event_dt > max_time:

            rejected_rows.append({
                "id": row["id"],
                "reasonCodes": ["OUT_OF_WINDOW"]
            })

            continue

        policy_retained.append(row)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train = []
    validation = []
    test = []

    for row in policy_retained:

        split = split_for_entity(row["entity"])

        if split == "train":
            train.append(row)
        elif split == "validation":
            validation.append(row)
        else:
            test.append(row)

    # --------------------------------------------------------
    # Train contamination
    # --------------------------------------------------------

    train_word_sets = [
        word_set(row["text"])
        for row in train
    ]

    def contamination_filter(rows):

        kept = []

        for row in rows:

            row_words = word_set(row["text"])

            contaminated = False

            for train_words in train_word_sets:

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
                kept.append(row)

        return kept

    validation = contamination_filter(validation)
    test = contamination_filter(test)

    # --------------------------------------------------------
    # Deterministic row sorting
    # --------------------------------------------------------

    def sort_rows(rows):
        return sorted(
            rows,
            key=lambda row: (
                utf8(row["id"]),
                utf8(compact_json(row))
            )
        )

    train = sort_rows(train)
    validation = sort_rows(validation)
    test = sort_rows(test)

    # --------------------------------------------------------
    # Exact JSONL serialization and SHA-256
    # --------------------------------------------------------

    def split_digest(rows):

        if not rows:
            data = b""
        else:
            lines = [
                compact_json({
                    "id": row["id"],
                    "entity": row["entity"],
                    "eventTime": row["eventTime"],
                    "revision": row["revision"],
                    "text": row["text"],
                })
                for row in rows
            ]

            data = (
                "\n".join(lines) + "\n"
            ).encode("utf-8")

        return hashlib.sha256(data).hexdigest()

    digests = {
        "train": split_digest(train),
        "validation": split_digest(validation),
        "test": split_digest(test),
    }

    # --------------------------------------------------------
    # Rejected rows:
    # merge same ID and sort reason codes.
    # --------------------------------------------------------

    rejected_by_id = {}

    for item in rejected_rows:

        row_id = item["id"]

        if row_id not in rejected_by_id:
            rejected_by_id[row_id] = set()

        for code in item["reasonCodes"]:
            rejected_by_id[row_id].add(code)

    rejected_rows_final = [
        {
            "id": row_id,
            "reasonCodes": sorted_codes(codes)
        }
        for row_id, codes in rejected_by_id.items()
    ]

    rejected_rows_final.sort(
        key=lambda item: (
            utf8(item["id"]),
            utf8(compact_json(item))
        )
    )

    # --------------------------------------------------------
    # Rejected object sorting
    # --------------------------------------------------------

    rejected_objects.sort(
        key=lambda item: (
            utf8(item["uri"]) if isinstance(item["uri"], str)
            else b"",
            utf8(compact_json(item))
        )
    )

    # --------------------------------------------------------
    # Lineage
    # --------------------------------------------------------

    lineage = []

    for obj in accepted_objects:

        lineage.append({
            "uri": obj["uri"],
            "generation": obj["generation"],
            "crc32c": obj["crc32c"],
            "schemaId": obj["schemaId"],
        })

    lineage.sort(
        key=lambda item: (
            utf8(item["uri"]),
            utf8(compact_json(item))
        )
    )

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    response = {
        "splits": {
            "train": train,
            "validation": validation,
            "test": test,
        },
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows_final,
        "digests": digests,
        "lineage": lineage,
    }

    return app.response_class(
        response=compact_json(response),
        status=200,
        mimetype="application/json"
    )


# ============================================================
# Simple root endpoint
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return "JSONL corpus service is running"


# ============================================================
# Local execution
# ============================================================

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 8080))

    app.run(
        host="0.0.0.0",
        port=port
    )