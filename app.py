import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone, timedelta
import os
import threading

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
# Q2 - Leakage-Safe BigQuery ML Experiment
# ============================================================

BQML_RUNS = {}
BQML_LOCK = threading.Lock()
BQML_SAFE_MAX = 2**53 - 1


def bqml_safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= BQML_SAFE_MAX
    )


def bqml_finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def bqml_unit(value):
    return bqml_finite(value) and 0 <= float(value) <= 1


def bqml_digest(value):
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def bqml_row_valid(row):
    if not isinstance(row, dict):
        return False

    if set(row) != {
        "id", "entity", "eventTime", "predictionTime",
        "version", "split", "features"
    }:
        return False

    if not isinstance(row["id"], str):
        return False
    if not isinstance(row["entity"], str):
        return False
    if not isinstance(row["eventTime"], str):
        return False
    if not isinstance(row["predictionTime"], str):
        return False
    if not bqml_safe_integer(row["version"]):
        return False
    if row["split"] not in ("TRAIN", "EVAL"):
        return False
    if not isinstance(row["features"], dict):
        return False

    try:
        parse_timestamp(row["eventTime"])
        parse_timestamp(row["predictionTime"])
    except Exception:
        return False

    for name, feature in row["features"].items():
        if not isinstance(name, str):
            return False
        if not isinstance(feature, dict):
            return False
        if set(feature) != {"value", "availableAt"}:
            return False
        # value is opaque JSON data. Do not execute or interpret it.
        if not isinstance(feature["availableAt"], str):
            return False
        try:
            parse_timestamp(feature["availableAt"])
        except Exception:
            return False

    return True


def bqml_deduplicate(rows):
    groups = {}
    for row in rows:
        key = (row["entity"], parse_timestamp(row["eventTime"]))
        groups.setdefault(key, []).append(row)

    retained = []
    for group in groups.values():
        retained.append(min(
            group,
            key=lambda r: (-r["version"], utf8(r["id"]))
        ))
    return retained


def bqml_features(rows, forbidden):
    if not rows:
        return []

    common = set(rows[0]["features"])
    for row in rows[1:]:
        common.intersection_update(row["features"])

    result = []
    for name in common:
        if name in forbidden:
            continue

        safe_for_all = True
        for row in rows:
            available = parse_timestamp(
                row["features"][name]["availableAt"]
            )
            prediction = parse_timestamp(row["predictionTime"])
            if available > prediction:
                safe_for_all = False
                break

        if safe_for_all:
            result.append(name)

    return sorted(result, key=utf8)


def bqml_trial_shape_valid(trial):
    if not isinstance(trial, dict):
        return False
    if set(trial) != {"trialId", "status", "evalMetric"}:
        return False
    if not bqml_safe_integer(trial["trialId"]):
        return False
    if trial["status"] not in ("SUCCEEDED", "FAILED"):
        return False
    return True


def bqml_trial_eligible(trial):
    return (
        trial["status"] == "SUCCEEDED"
        and bqml_finite(trial["evalMetric"])
    )


def bqml_fingerprint(body):
    # Canonical fingerprint for replay/conflict detection. Object
    # member ordering is not semantically significant.
    raw = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def bqml_selection(body):
    required = {
        "phase", "runId", "forbiddenFeatures",
        "numTrialsLimit", "rows", "trials"
    }

    if not isinstance(body, dict) or set(body) != required:
        return None, None

    codes = []
    run_id = body["runId"]

    if not isinstance(run_id, str) or not run_id or len(run_id) > 128:
        codes.append("INVALID_INPUT")

    forbidden = body["forbiddenFeatures"]
    if not isinstance(forbidden, list) or any(
        not isinstance(x, str) for x in forbidden
    ):
        codes.append("INVALID_INPUT")
        forbidden = []

    limit = body["numTrialsLimit"]
    if not bqml_safe_integer(limit) or limit <= 0:
        codes.append("INVALID_INPUT")

    rows = body["rows"]
    if not isinstance(rows, list) or not rows:
        codes.append("INVALID_INPUT")

    trials = body["trials"]
    if not isinstance(trials, list):
        codes.append("INVALID_INPUT")
        trials = []

    valid_rows = []
    seen_rows = set()
    if isinstance(rows, list):
        for row in rows:
            if not bqml_row_valid(row):
                codes.append("INVALID_INPUT")
                continue
            if row["id"] in seen_rows:
                codes.append("INVALID_INPUT")
                continue
            seen_rows.add(row["id"])
            valid_rows.append(row)

    eligible_trials = []
    seen_trials = set()
    for trial in trials:
        if not bqml_trial_shape_valid(trial):
            codes.append("INVALID_INPUT")
            continue
        if trial["trialId"] in seen_trials:
            codes.append("INVALID_INPUT")
            continue
        seen_trials.add(trial["trialId"])
        if bqml_trial_eligible(trial):
            eligible_trials.append(trial)

    if bqml_safe_integer(limit) and len(trials) > limit:
        codes.append("TRIAL_LIMIT_EXCEEDED")

    codes = sorted_codes(codes)
    fingerprint = bqml_fingerprint(body)

    if codes:
        return ({
            "runId": run_id if isinstance(run_id, str) else None,
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": codes
        }, fingerprint)

    if not eligible_trials:
        return ({
            "runId": run_id,
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": ["NO_SUCCESSFUL_TRIAL"]
        }, fingerprint)

    retained = bqml_deduplicate(valid_rows)

    feature_names = bqml_features(retained, set(forbidden))
    train_ids = sorted(
        [r["id"] for r in retained if r["split"] == "TRAIN"],
        key=utf8
    )
    eval_ids = sorted(
        [r["id"] for r in retained if r["split"] == "EVAL"],
        key=utf8
    )

    selected = min(
        eligible_trials,
        key=lambda t: (-float(t["evalMetric"]), t["trialId"])
    )

    digest_object = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names
    }
    dataset_digest = hashlib.sha256(
        compact_json(digest_object).encode("utf-8")
    ).hexdigest()

    response = {
        "runId": run_id,
        "selectedTrialId": selected["trialId"],
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": dataset_digest,
        "reasonCodes": []
    }
    return response, fingerprint


def bqml_test_row_valid(row):
    if not isinstance(row, dict):
        return False
    if set(row) != {"label", "prediction", "slice"}:
        return False
    if (
        isinstance(row["label"], bool)
        or not isinstance(row["label"], int)
        or row["label"] not in (0, 1)
    ):
        return False
    if (
        isinstance(row["prediction"], bool)
        or not isinstance(row["prediction"], int)
        or row["prediction"] not in (0, 1)
    ):
        return False
    if not isinstance(row["slice"], str) or row["slice"] == "":
        return False
    return True


def bqml_evaluation(body):
    required = {
        "phase", "runId", "selectedTrialId", "datasetDigest",
        "metricFloor", "requiredSlices", "rows",
        "bytesProcessed", "maxBytes"
    }

    if not isinstance(body, dict) or set(body) != required:
        return {
            "runId": body.get("runId") if isinstance(body, dict) else None,
            "selectedTrialId": (
                body.get("selectedTrialId") if isinstance(body, dict) else None
            ),
            "datasetDigest": (
                body.get("datasetDigest") if isinstance(body, dict) else None
            ),
            "testMetric": None,
            "criticalSlicePass": False,
            "decision": "reject",
            "bytesProcessed": (
                body.get("bytesProcessed", 0) if isinstance(body, dict) else 0
            ),
            "reasonCodes": ["INVALID_INPUT"]
        }

    run_id = body["runId"]
    selected_id = body["selectedTrialId"]
    digest = body["datasetDigest"]
    metric_floor = body["metricFloor"]
    required_slices = body["requiredSlices"]
    rows = body["rows"]
    bytes_processed = body["bytesProcessed"]
    max_bytes = body["maxBytes"]

    codes = []

    if not isinstance(run_id, str) or not run_id or len(run_id) > 128:
        codes.append("INVALID_INPUT")
    if not bqml_safe_integer(selected_id):
        codes.append("INVALID_INPUT")
    if not bqml_digest(digest):
        codes.append("INVALID_INPUT")
    if not bqml_unit(metric_floor):
        codes.append("INVALID_INPUT")

    if not isinstance(required_slices, dict):
        codes.append("INVALID_INPUT")
        required_slices = {}
    else:
        for name, floor in required_slices.items():
            if not isinstance(name, str) or not bqml_unit(floor):
                codes.append("INVALID_INPUT")

    if not isinstance(rows, list):
        codes.append("INVALID_INPUT")
    if not bqml_safe_integer(bytes_processed):
        codes.append("INVALID_INPUT")
    if not bqml_safe_integer(max_bytes):
        codes.append("INVALID_INPUT")

    with BQML_LOCK:
        stored = BQML_RUNS.get(run_id) if isinstance(run_id, str) else None

    if (
        stored is None
        or stored["reasonCodes"]
        or stored["selectedTrialId"] is None
        or stored["datasetDigest"] is None
        or stored["selectedTrialId"] != selected_id
        or stored["datasetDigest"] != digest
    ):
        codes.append("INVALID_LINEAGE")

    valid_rows = []
    every_valid = True
    if isinstance(rows, list):
        for row in rows:
            if not bqml_test_row_valid(row):
                every_valid = False
            else:
                valid_rows.append(row)
        if not every_valid:
            codes.append("INVALID_TEST_ROW")

    skip_metrics = (
        not isinstance(rows, list)
        or len(rows) == 0
        or not every_valid
    )

    test_metric = None
    critical_slice_pass = not (
        "INVALID_INPUT" in codes
        or "INVALID_LINEAGE" in codes
        or "INVALID_TEST_ROW" in codes
        or skip_metrics
    )

    if not skip_metrics:
        correct = sum(
            row["label"] == row["prediction"]
            for row in valid_rows
        )
        test_metric = round(correct / len(valid_rows), 12)

        if test_metric < float(metric_floor):
            codes.append("AGGREGATE_FLOOR")

        groups = {}
        for row in valid_rows:
            groups.setdefault(row["slice"], []).append(row)

        for name, floor in required_slices.items():
            if name not in groups:
                codes.append(f"MISSING_SLICE:{name}")
                critical_slice_pass = False
                continue

            slice_rows = groups[name]
            slice_correct = sum(
                row["label"] == row["prediction"]
                for row in slice_rows
            )
            accuracy = round(slice_correct / len(slice_rows), 12)

            if accuracy < float(floor):
                codes.append(f"SLICE_FLOOR:{name}")
                critical_slice_pass = False

    # Cost gate remains active even if test rows are empty/invalid.
    if (
        bqml_safe_integer(bytes_processed)
        and bqml_safe_integer(max_bytes)
        and bytes_processed > max_bytes
    ):
        codes.append("BYTE_LIMIT")

    codes = sorted_codes(codes)

    return {
        "runId": run_id,
        "selectedTrialId": selected_id,
        "datasetDigest": digest,
        "testMetric": test_metric,
        "criticalSlicePass": bool(critical_slice_pass),
        "decision": "admit" if not codes else "reject",
        "bytesProcessed": bytes_processed,
        "reasonCodes": codes
    }


@app.route("/bqml", methods=["POST"])
def bqml():

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

    phase = body.get("phase")
    if phase not in ("select", "evaluate"):
        return app.response_class(
            response='{"error":"INVALID_INPUT"}',
            status=400,
            mimetype="application/json"
        )

    if phase == "select":
        run_id = body.get("runId")
        fingerprint = bqml_fingerprint(body)

        if isinstance(run_id, str):
            with BQML_LOCK:
                existing = BQML_RUNS.get(run_id)

            if existing is not None:
                if existing["fingerprint"] == fingerprint:
                    return app.response_class(
                        response=existing["response_json"],
                        status=200,
                        mimetype="application/json"
                    )
                return app.response_class(
                    response='{"error":"RUN_ID_CONFLICT"}',
                    status=409,
                    mimetype="application/json"
                )

        response, fingerprint = bqml_selection(body)

        if response is None:
            return app.response_class(
                response='{"error":"INVALID_INPUT"}',
                status=400,
                mimetype="application/json"
            )

        response_json = compact_json(response)
        response_run_id = response["runId"]

        if isinstance(response_run_id, str):
            with BQML_LOCK:
                existing = BQML_RUNS.get(response_run_id)

                if existing is not None:
                    if existing["fingerprint"] == fingerprint:
                        return app.response_class(
                            response=existing["response_json"],
                            status=200,
                            mimetype="application/json"
                        )
                    return app.response_class(
                        response='{"error":"RUN_ID_CONFLICT"}',
                        status=409,
                        mimetype="application/json"
                    )

                BQML_RUNS[response_run_id] = {
                    "fingerprint": fingerprint,
                    "response": response,
                    "response_json": response_json,
                    "selectedTrialId": response["selectedTrialId"],
                    "datasetDigest": response["datasetDigest"],
                    "reasonCodes": response["reasonCodes"]
                }

        return app.response_class(
            response=response_json,
            status=200,
            mimetype="application/json"
        )

    response = bqml_evaluation(body)
    return app.response_class(
        response=compact_json(response),
        status=200,
        mimetype="application/json"
    )


# ============================================================
# Q3 - MLflow Model Promotion Gate
# Endpoint: POST /promote
# ============================================================

PROMOTION_LOCK = threading.Lock()
PROMOTION_REPLAYS = {}
PROMOTION_ALIAS_VERSION = None


def promote_safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= 2**53 - 1
    )


def promote_finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def promote_unit(value):
    return (
        promote_finite(value)
        and 0 <= float(value) <= 1
    )


def promote_version(value):
    # Positive safe-integer, canonical decimal string.
    if not isinstance(value, str):
        return False
    if not re.fullmatch(r"[1-9][0-9]*", value):
        return False
    try:
        n = int(value)
    except Exception:
        return False
    return 1 <= n <= 2**53 - 1 and str(n) == value


def promote_timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        return parse_timestamp(value)
    except Exception:
        return None


def promote_policy_valid(policy):
    if not isinstance(policy, dict):
        return False

    required = {
        "datasetDigest",
        "schemaDigest",
        "maxAgeSeconds",
        "accuracyFloor",
        "requiredSlices",
        "maxLatencyMs",
        "maxSizeBytes",
        "minImprovement",
    }

    if not required.issubset(policy.keys()):
        return False

    if (
        not isinstance(policy["datasetDigest"], str)
        or not policy["datasetDigest"]
    ):
        return False

    if (
        not isinstance(policy["schemaDigest"], str)
        or not policy["schemaDigest"]
    ):
        return False

    if (
        not promote_safe_integer(policy["maxAgeSeconds"])
    ):
        return False

    if not promote_unit(policy["accuracyFloor"]):
        return False

    required_slices = policy["requiredSlices"]
    if not isinstance(required_slices, dict):
        return False

    for name, floor in required_slices.items():
        if not isinstance(name, str):
            return False
        if not promote_unit(floor):
            return False

    if not promote_finite(policy["maxLatencyMs"]):
        return False
    if float(policy["maxLatencyMs"]) < 0:
        return False

    if not promote_safe_integer(policy["maxSizeBytes"]):
        return False

    if not promote_unit(policy["minImprovement"]):
        return False

    return True


def promote_evaluation_shape_valid(evaluation):
    if not isinstance(evaluation, dict):
        return False

    required = {
        "createdAt",
        "artifactDigest",
        "datasetDigest",
        "schemaDigest",
        "accuracy",
        "latencyMs",
        "sizeBytes",
        "slices",
    }

    return required.issubset(evaluation.keys())


def promote_failed_codes_add(failed, version, codes):
    if version not in failed:
        failed[version] = []
    failed[version].extend(codes)


def promote_failed_codes_normalize(failed):
    for version in list(failed.keys()):
        failed[version] = sorted_codes(failed[version])


def promote_fingerprint(body):
    # Request replay identity is independent of JSON member ordering.
    return hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


def promote_evaluate_version(version_obj, policy, as_of):
    """
    Return all independently applicable gate failures for one
    otherwise structurally valid version.

    Mutable tags/descriptions are deliberately ignored.
    """

    failures = []

    if "evaluation" not in version_obj or version_obj["evaluation"] is None:
        return ["MISSING_EVALUATION"]

    evaluation = version_obj["evaluation"]

    if not promote_evaluation_shape_valid(evaluation):
        # A present but unusable evaluation has no dedicated
        # schema code in this contract. Treat it as missing
        # evidence.
        return ["MISSING_EVALUATION"]

    created = promote_timestamp(evaluation["createdAt"])

    if created is None:
        failures.append("INVALID_TIMESTAMP")

    # Numeric evidence checks.
    accuracy = evaluation["accuracy"]
    latency = evaluation["latencyMs"]
    size = evaluation["sizeBytes"]

    if not promote_finite(accuracy):
        failures.append("NON_FINITE")
    elif not promote_unit(accuracy):
        failures.append("METRIC_RANGE")

    if not promote_finite(latency):
        failures.append("NON_FINITE")
    elif float(latency) < 0:
        failures.append("METRIC_RANGE")

    if not promote_safe_integer(size):
        failures.append("NON_FINITE")
    # sizeBytes is specified as a non-negative safe integer;
    # an invalid type/value is a metric/evidence failure.
    elif size < 0:
        failures.append("METRIC_RANGE")

    # Evaluation timestamp gates.
    if created is not None:
        if created > as_of:
            failures.append("FUTURE_EVALUATION")
        else:
            lower = as_of - timedelta(
                seconds=policy["maxAgeSeconds"]
            )
            if created < lower:
                failures.append("STALE_EVALUATION")

    # Registered artifact must match evaluation evidence.
    if evaluation["artifactDigest"] != version_obj.get("artifactDigest"):
        failures.append("ARTIFACT_MISMATCH")

    if evaluation["datasetDigest"] != policy["datasetDigest"]:
        failures.append("DATASET_MISMATCH")

    if evaluation["schemaDigest"] != policy["schemaDigest"]:
        failures.append("SCHEMA_MISMATCH")

    # Aggregate gates. Do not add these when the corresponding
    # numeric value is unusable; NON_FINITE/METRIC_RANGE is the
    # applicable evidence failure.
    if promote_unit(accuracy):
        if float(accuracy) < float(policy["accuracyFloor"]):
            failures.append("ACCURACY_FLOOR")

    if promote_finite(latency) and float(latency) >= 0:
        if float(latency) > float(policy["maxLatencyMs"]):
            failures.append("LATENCY_LIMIT")

    if promote_safe_integer(size):
        if size > policy["maxSizeBytes"]:
            failures.append("SIZE_LIMIT")

    # Required slices.
    slices = evaluation["slices"]

    if not isinstance(slices, dict):
        for name in policy["requiredSlices"]:
            failures.append(f"MISSING_SLICE:{name}")
    else:
        for name, floor in policy["requiredSlices"].items():

            if name not in slices:
                failures.append(f"MISSING_SLICE:{name}")
                continue

            value = slices[name]

            if not promote_finite(value):
                failures.append(f"SLICE_RANGE:{name}")
                continue

            if not promote_unit(value):
                failures.append(f"SLICE_RANGE:{name}")
                continue

            if float(value) < float(floor):
                failures.append(f"SLICE_FLOOR:{name}")

    return sorted_codes(failures)


def promote_process(body):
    """
    Process a valid /promote request.
    """

    as_of = promote_timestamp(body["asOf"])
    policy = body["policy"]
    versions = body["versions"]
    champion_version = body["championVersion"]

    failed = {}
    valid_objects = []
    seen_versions = set()

    # --------------------------------------------------------
    # First pass: reject noncanonical/duplicate versions BEFORE
    # constructing the lookup map.
    # --------------------------------------------------------

    for obj in versions:

        if not isinstance(obj, dict):
            # No version key available to attach to.
            continue

        version = obj.get("version")

        if not promote_version(version):
            if isinstance(version, str):
                promote_failed_codes_add(
                    failed,
                    version,
                    ["INVALID_VERSION"]
                )
            continue

        if version in seen_versions:
            promote_failed_codes_add(
                failed,
                version,
                ["DUPLICATE_VERSION"]
            )
            continue

        seen_versions.add(version)

        valid_objects.append(obj)

    # A canonical version that has appeared more than once is
    # unusable. Remove it from the lookup/candidate set.
    duplicate_versions = {
        version
        for version, codes in failed.items()
        if "DUPLICATE_VERSION" in codes
    }

    valid_objects = [
        obj
        for obj in valid_objects
        if obj["version"] not in duplicate_versions
    ]

    # --------------------------------------------------------
    # Every listed canonical version receives its gate list.
    # --------------------------------------------------------

    if not promote_policy_valid(policy):
        for obj in valid_objects:
            promote_failed_codes_add(
                failed,
                obj["version"],
                ["INVALID_POLICY"]
            )

        for version in list(failed.keys()):
            if (
                promote_version(version)
                and version not in duplicate_versions
            ):
                promote_failed_codes_add(
                    failed,
                    version,
                    ["INVALID_POLICY"]
                )

        promote_failed_codes_normalize(failed)

        return {
            "action": "block",
            "championVersion": champion_version,
            "selectedVersion": None,
            "eligibleVersions": [],
            "failedGates": failed,
            "aliasMutation": None,
            "evidence": None,
        }

    # --------------------------------------------------------
    # Build lookup only AFTER invalid/duplicate rejection.
    # --------------------------------------------------------

    lookup = {
        obj["version"]: obj
        for obj in valid_objects
    }

    eligible = []

    for obj in valid_objects:

        version = obj["version"]

        failures = promote_evaluate_version(
            obj,
            policy,
            as_of
        )

        promote_failed_codes_add(
            failed,
            version,
            failures
        )

        if not failures:
            eligible.append(obj)

    # --------------------------------------------------------
    # Champion evidence must be valid.
    # --------------------------------------------------------

    champion_obj = lookup.get(
        champion_version
    )

    champion_valid = (
        champion_obj is not None
        and champion_version not in duplicate_versions
        and not any(
            failed.get(champion_version, [])
        )
    )

    # If champion itself is not a valid listed version, make the
    # reason visible when possible.
    if champion_obj is None:
        if not promote_version(champion_version):
            promote_failed_codes_add(
                failed,
                champion_version,
                ["INVALID_VERSION"]
            )
        else:
            promote_failed_codes_add(
                failed,
                champion_version,
                ["MISSING_EVALUATION"]
            )

        promote_failed_codes_normalize(failed)
        return {
            "action": "block",
            "championVersion": champion_version,
            "selectedVersion": None,
            "eligibleVersions": sorted(
                [v["version"] for v in eligible],
                key=lambda x: int(x)
            ),
            "failedGates": failed,
            "aliasMutation": None,
            "evidence": None,
        }

    if not champion_valid:
        promote_failed_codes_normalize(failed)
        return {
            "action": "block",
            "championVersion": champion_version,
            "selectedVersion": None,
            "eligibleVersions": sorted(
                [v["version"] for v in eligible],
                key=lambda x: int(x)
            ),
            "failedGates": failed,
            "aliasMutation": None,
            "evidence": None,
        }

    # --------------------------------------------------------
    # Rank eligible versions:
    # accuracy DESC, latency ASC, size ASC, numeric version ASC
    # --------------------------------------------------------

    eligible.sort(
        key=lambda obj: (
            -float(obj["evaluation"]["accuracy"]),
            float(obj["evaluation"]["latencyMs"]),
            obj["evaluation"]["sizeBytes"],
            int(obj["version"])
        )
    )

    eligible_versions = [
        obj["version"]
        for obj in sorted(
            eligible,
            key=lambda obj: int(obj["version"])
        )
    ]

    # Best eligible challenger is the top-ranked eligible version
    # other than the champion.
    challengers = [
        obj
        for obj in eligible
        if obj["version"] != champion_version
    ]

    if not challengers:
        promote_failed_codes_normalize(failed)

        return {
            "action": "retain",
            "championVersion": champion_version,
            "selectedVersion": champion_version,
            "eligibleVersions": eligible_versions,
            "failedGates": failed,
            "aliasMutation": None,
            "evidence": champion_obj["evaluation"],
        }

    challenger = challengers[0]

    improvement = round(
        float(challenger["evaluation"]["accuracy"])
        - float(champion_obj["evaluation"]["accuracy"]),
        12
    )

    if improvement >= float(policy["minImprovement"]):

        result = {
            "action": "promote",
            "championVersion": champion_version,
            "selectedVersion": challenger["version"],
            "eligibleVersions": eligible_versions,
            "failedGates": failed,
            "aliasMutation": {
                "alias": "champion",
                "version": challenger["version"]
            },
            "evidence": challenger["evaluation"],
        }

        # Persist alias mutation and exact replay response.
        with PROMOTION_LOCK:
            global PROMOTION_ALIAS_VERSION
            PROMOTION_ALIAS_VERSION = challenger["version"]

        promote_failed_codes_normalize(
            result["failedGates"]
        )

        return result

    promote_failed_codes_normalize(failed)

    return {
        "action": "retain",
        "championVersion": champion_version,
        "selectedVersion": champion_version,
        "eligibleVersions": eligible_versions,
        "failedGates": failed,
        "aliasMutation": None,
        "evidence": champion_obj["evaluation"],
    }


@app.route("/promote", methods=["POST"])
def promote():

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

    # --------------------------------------------------------
    # Explicit HTTP-400 contract cases.
    # --------------------------------------------------------

    if not isinstance(body, dict):
        return app.response_class(
            response='{"error":"INVALID_INPUT"}',
            status=400,
            mimetype="application/json"
        )

    if not isinstance(body.get("championVersion"), str):
        return app.response_class(
            response='{"error":"INVALID_INPUT"}',
            status=400,
            mimetype="application/json"
        )

    if "policy" not in body:
        return app.response_class(
            response='{"error":"INVALID_INPUT"}',
            status=400,
            mimetype="application/json"
        )

    if "versions" not in body or not isinstance(
        body["versions"], list
    ):
        return app.response_class(
            response='{"error":"INVALID_INPUT"}',
            status=400,
            mimetype="application/json"
        )

    if "asOf" not in body:
        return app.response_class(
            response='{"error":"INVALID_INPUT"}',
            status=400,
            mimetype="application/json"
        )

    # Other malformed top-level requests are represented by
    # INVALID_POLICY / INVALID_VERSION where applicable, but a
    # missing core request member is an invalid request.
    required = {
        "asOf",
        "championVersion",
        "policy",
        "versions"
    }

    if not required.issubset(body.keys()):
        return app.response_class(
            response='{"error":"INVALID_INPUT"}',
            status=400,
            mimetype="application/json"
        )

    # --------------------------------------------------------
    # Validate asOf before processing evidence.
    # --------------------------------------------------------

    if promote_timestamp(body["asOf"]) is None:
        return app.response_class(
            response='{"error":"INVALID_INPUT"}',
            status=400,
            mimetype="application/json"
        )

    # --------------------------------------------------------
    # Deterministic replay.
    # --------------------------------------------------------

    fingerprint = promote_fingerprint(body)

    with PROMOTION_LOCK:
        replay = PROMOTION_REPLAYS.get(
            fingerprint
        )

    if replay is not None:
        return app.response_class(
            response=replay,
            status=200,
            mimetype="application/json"
        )

    result = promote_process(body)
    response_json = compact_json(result)

    with PROMOTION_LOCK:
        PROMOTION_REPLAYS[fingerprint] = response_json

    return app.response_class(
        response=response_json,
        status=200,
        mimetype="application/json"
    )



# ============================================================
# Q4 - Deterministic PEFT Adaptation Gate
# Endpoint: POST /adapt
# ============================================================

ADAPT_INTERVENTIONS = (
    "prompt_only",
    "retrieval",
    "lora",
    "qlora",
)

ADAPT_CHOOSE_CODES = {
    "INVALID_INPUT",
    "UNAVAILABLE",
    "QUALITY_FLOOR",
    "FRESHNESS_REQUIRED",
    "LATENCY_LIMIT",
    "MEMORY_LIMIT",
    "DATA_LIMIT",
    "COST_LIMIT",
}

ADAPT_REPAIR_CODES = {
    "INVALID_TOKEN",
    "INVALID_PARAMETER",
    "CHAT_TEMPLATE_COUNT",
    "INFERENCE_MODE",
    "FULL_MODEL_ARTIFACT",
    "ADAPTER_FILE_SET",
    "INCOMPLETE_CHECKPOINT",
    "MUTABLE_BASE_REVISION",
    "LINEAGE_MISMATCH",
    "EFFECTIVE_BATCH_MISMATCH",
    "EVAL_LEAKAGE",
    "EVAL_DROPOUT_ACTIVE",
    "RESUME_DIVERGENCE",
}


# ============================================================
# Q4 helpers
# ============================================================

def adapt_safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= 9007199254740991
    )


def adapt_finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def adapt_nonnegative_finite(value):
    return (
        adapt_finite(value)
        and float(value) >= 0
    )


def adapt_unit_interval(value):
    return (
        adapt_finite(value)
        and 0 <= float(value) <= 1
    )


def adapt_hex(value, length):
    return (
        isinstance(value, str)
        and re.fullmatch(
            rf"[0-9a-f]{{{length}}}",
            value
        ) is not None
    )


def adapt_unique_strings(value, nonempty=False):
    if not isinstance(value, list):
        return False

    if nonempty and len(value) == 0:
        return False

    if any(not isinstance(x, str) for x in value):
        return False

    if len(set(value)) != len(value):
        return False

    return True


def adapt_sorted_utf8(values):
    return sorted(values, key=utf8)


# ============================================================
# CHOOSE
# ============================================================

def adapt_validate_candidate(candidate, policy):
    """
    Return:
        (valid_structure, total_cost, reason_codes)

    A candidate can have multiple applicable gate failures.
    """

    if not isinstance(candidate, dict):
        return (
            False,
            None,
            ["INVALID_INPUT"]
        )

    required = {
        "name",
        "available",
        "quality",
        "freshness",
        "latencyMs",
        "memoryMb",
        "labeledExamples",
        "oneTimeCost",
        "recurringCost",
    }

    if set(candidate.keys()) != required:
        return (
            False,
            None,
            ["INVALID_INPUT"]
        )

    codes = []

    # --------------------------------------------------------
    # Basic candidate validation
    # --------------------------------------------------------

    if not isinstance(candidate["name"], str):
        codes.append("INVALID_INPUT")

    if not isinstance(candidate["available"], bool):
        codes.append("INVALID_INPUT")

    if not adapt_unit_interval(candidate["quality"]):
        codes.append("INVALID_INPUT")

    if not isinstance(candidate["freshness"], bool):
        codes.append("INVALID_INPUT")

    if not adapt_nonnegative_finite(
        candidate["latencyMs"]
    ):
        codes.append("INVALID_INPUT")

    if not adapt_nonnegative_finite(
        candidate["memoryMb"]
    ):
        codes.append("INVALID_INPUT")

    if not adapt_safe_integer(
        candidate["labeledExamples"]
    ):
        codes.append("INVALID_INPUT")

    if not adapt_nonnegative_finite(
        candidate["oneTimeCost"]
    ):
        codes.append("INVALID_INPUT")

    if not adapt_nonnegative_finite(
        candidate["recurringCost"]
    ):
        codes.append("INVALID_INPUT")

    # If the candidate itself is malformed, don't invent
    # additional gate failures from unusable values.
    if "INVALID_INPUT" in codes:
        return (
            False,
            None,
            sorted_codes(codes)
        )

    # --------------------------------------------------------
    # Total cost
    # --------------------------------------------------------

    total_cost = round(
        float(candidate["oneTimeCost"])
        + float(policy["horizonRequests"])
        * float(candidate["recurringCost"]),
        12
    )

    # --------------------------------------------------------
    # Eligibility gates
    # --------------------------------------------------------

    if not candidate["available"]:
        codes.append("UNAVAILABLE")

    if (
        candidate["quality"]
        < policy["minQuality"]
    ):
        codes.append("QUALITY_FLOOR")

    if (
        policy["freshnessRequired"]
        and not candidate["freshness"]
    ):
        codes.append("FRESHNESS_REQUIRED")

    if (
        candidate["latencyMs"]
        > policy["maxLatencyMs"]
    ):
        codes.append("LATENCY_LIMIT")

    if (
        candidate["memoryMb"]
        > policy["maxMemoryMb"]
    ):
        codes.append("MEMORY_LIMIT")

    if (
        candidate["labeledExamples"]
        > policy["maxLabeledExamples"]
    ):
        codes.append("DATA_LIMIT")

    if (
        total_cost
        > policy["maxTotalCost"]
    ):
        codes.append("COST_LIMIT")

    return (
        True,
        total_cost,
        sorted_codes(codes)
    )


def adapt_choose(body):
    required = {
        "operation",
        "policy",
        "candidates",
    }

    if (
        not isinstance(body, dict)
        or set(body.keys()) != required
        or body.get("operation") != "choose"
    ):
        return None

    policy = body["policy"]

    if not isinstance(policy, dict):
        return None

    policy_required = {
        "minQuality",
        "freshnessRequired",
        "maxLatencyMs",
        "maxMemoryMb",
        "maxLabeledExamples",
        "maxTotalCost",
        "horizonRequests",
    }

    if set(policy.keys()) != policy_required:
        return None

    policy_invalid = False

    if not adapt_unit_interval(
        policy["minQuality"]
    ):
        policy_invalid = True

    if not isinstance(
        policy["freshnessRequired"],
        bool
    ):
        policy_invalid = True

    if not adapt_nonnegative_finite(
        policy["maxLatencyMs"]
    ):
        policy_invalid = True

    if not adapt_nonnegative_finite(
        policy["maxMemoryMb"]
    ):
        policy_invalid = True

    if not adapt_safe_integer(
        policy["maxLabeledExamples"]
    ):
        policy_invalid = True

    if not adapt_nonnegative_finite(
        policy["maxTotalCost"]
    ):
        policy_invalid = True

    if (
        not adapt_safe_integer(
            policy["horizonRequests"]
        )
    ):
        policy_invalid = True

    if policy_invalid:
        return None

    candidates = body["candidates"]

    if not isinstance(candidates, list):
        return None

    # Exactly one candidate for each intervention.
    if len(candidates) != 4:
        return None

    names = [
        candidate.get("name")
        if isinstance(candidate, dict)
        else None
        for candidate in candidates
    ]

    if (
        any(
            name not in ADAPT_INTERVENTIONS
            for name in names
        )
        or len(set(names)) != 4
    ):
        return None

    by_name = {
        candidate["name"]: candidate
        for candidate in candidates
    }

    eligible = []
    total_costs = {}
    reason_codes = {}

    for name in ADAPT_INTERVENTIONS:

        candidate = by_name[name]

        valid, total_cost, codes = (
            adapt_validate_candidate(
                candidate,
                policy
            )
        )

        if total_cost is None:
            total_costs[name] = None
        else:
            total_costs[name] = total_cost

        reason_codes[name] = sorted_codes(codes)

        if valid and not codes:
            eligible.append(name)

    selected = (
        eligible[0]
        if eligible
        else None
    )

    return {
        "selected": selected,
        "eligible": eligible,
        "totalCosts": total_costs,
        "reasonCodes": reason_codes,
    }


# ============================================================
# REPAIR - token/loss-mask validation
# ============================================================

def adapt_valid_token(token):
    if not isinstance(token, dict):
        return False

    if set(token.keys()) != {
        "id",
        "role",
        "padding",
        "text"
    }:
        return False

    if not adapt_safe_integer(token["id"]):
        return False

    if token["role"] not in (
        "system",
        "user",
        "assistant"
    ):
        return False

    if not isinstance(
        token["padding"],
        bool
    ):
        return False

    if not isinstance(
        token["text"],
        str
    ):
        return False

    return True


def adapt_make_labels(tokens):
    """
    Valid token list:
      unpadded assistant -> ID
      everything else    -> -100

    Any invalid token means ALL labels are -100.
    """

    if not isinstance(tokens, list):
        return (
            [-100] * (
                len(tokens)
                if isinstance(tokens, list)
                else 0
            ),
            False
        )

    valid = all(
        adapt_valid_token(token)
        for token in tokens
    )

    if not valid:
        return (
            [-100] * len(tokens),
            False
        )

    labels = []

    for token in tokens:

        if (
            token["role"] == "assistant"
            and not token["padding"]
        ):
            labels.append(token["id"])
        else:
            labels.append(-100)

    return labels, True


# ============================================================
# REPAIR - parameter validation
# ============================================================

def adapt_parameter_valid(parameter):
    if not isinstance(parameter, dict):
        return False

    if set(parameter.keys()) != {
        "name",
        "target",
        "numel"
    }:
        return False

    if not isinstance(
        parameter["name"],
        str
    ):
        return False

    if not isinstance(
        parameter["target"],
        str
    ):
        return False

    if (
        not adapt_safe_integer(
            parameter["numel"]
        )
        or parameter["numel"] <= 0
    ):
        return False

    return True


def adapt_is_lora_parameter(parameter, allowed_targets):
    return (
        parameter["target"] in allowed_targets
        and (
            parameter["name"].endswith(
                ".lora_A.weight"
            )
            or parameter["name"].endswith(
                ".lora_B.weight"
            )
        )
    )


def adapt_sum_numel(parameters):
    total = 0

    for parameter in parameters:
        total += parameter["numel"]

        if total > 9007199254740991:
            return None

    return total


# ============================================================
# REPAIR
# ============================================================

def adapt_repair(body):

    codes = []

    # --------------------------------------------------------
    # Tokens
    # --------------------------------------------------------

    tokens = body.get("tokens")

    labels, tokens_valid = adapt_make_labels(
        tokens
    )

    if not tokens_valid:
        codes.append("INVALID_TOKEN")

    # --------------------------------------------------------
    # Template application
    # --------------------------------------------------------

    if body.get("templateApplications") != 1:
        codes.append(
            "CHAT_TEMPLATE_COUNT"
        )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    parameters = body.get("parameters")
    allowed_targets = body.get("allowedTargets")

    parameters_valid = (
        isinstance(parameters, list)
        and isinstance(allowed_targets, list)
    )

    if not parameters_valid:
        codes.append("INVALID_PARAMETER")
        parameters = []
        allowed_targets = []

    else:

        if (
            len(set(
                parameter.get("name")
                for parameter in parameters
                if isinstance(parameter, dict)
            ))
            != len(parameters)
        ):
            codes.append("INVALID_PARAMETER")

        if not adapt_unique_strings(
            allowed_targets,
            nonempty=True
        ):
            codes.append("INVALID_PARAMETER")
            allowed_targets = []

        for parameter in parameters:
            if not adapt_parameter_valid(
                parameter
            ):
                codes.append(
                    "INVALID_PARAMETER"
                )

    # --------------------------------------------------------
    # Trainable LoRA parameters
    # --------------------------------------------------------

    trainable = []

    for parameter in parameters:

        if (
            adapt_parameter_valid(parameter)
            and adapt_is_lora_parameter(
                parameter,
                set(allowed_targets)
            )
        ):
            trainable.append(
                parameter
            )

    if not trainable:
        codes.append("INVALID_PARAMETER")

    trainable.sort(
        key=lambda parameter: utf8(
            parameter["name"]
        )
    )

    trainable_names = [
        parameter["name"]
        for parameter in trainable
    ]

    trainable_count = adapt_sum_numel(
        trainable
    )

    if trainable_count is None:
        codes.append("INVALID_PARAMETER")
        trainable_count = 0

    # --------------------------------------------------------
    # Inference / dropout
    # --------------------------------------------------------

    if body.get("inferenceMode") is not False:
        codes.append("INFERENCE_MODE")

    if body.get("dropoutActiveDuringEval") is not False:
        codes.append(
            "EVAL_DROPOUT_ACTIVE"
        )

    # --------------------------------------------------------
    # Train/eval IDs
    # --------------------------------------------------------

    train_ids = body.get("trainRowIds")
    eval_ids = body.get("evalRowIds")

    train_valid = adapt_unique_strings(
        train_ids,
        nonempty=True
    )

    eval_valid = adapt_unique_strings(
        eval_ids,
        nonempty=True
    )

    if not train_valid or not eval_valid:
        codes.append("EVAL_LEAKAGE")
        train_ids = (
            train_ids
            if isinstance(train_ids, list)
            else []
        )
        eval_ids = (
            eval_ids
            if isinstance(eval_ids, list)
            else []
        )
    else:

        if set(train_ids) & set(eval_ids):
            codes.append("EVAL_LEAKAGE")

    train_ids_sorted = adapt_sorted_utf8(
        train_ids
    )

    eval_ids_sorted = adapt_sorted_utf8(
        eval_ids
    )

    # --------------------------------------------------------
    # Adapter files
    # --------------------------------------------------------

    artifact_files = body.get(
        "artifactFiles"
    )

    expected_adapter_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]

    adapter_files_valid = (
        isinstance(artifact_files, list)
        and len(artifact_files) == 2
        and all(
            isinstance(x, str)
            for x in artifact_files
        )
        and sorted(
            artifact_files,
            key=utf8
        ) == sorted(
            expected_adapter_files,
            key=utf8
        )
    )

    if not adapter_files_valid:
        codes.append(
            "ADAPTER_FILE_SET"
        )

    adapter_files = adapt_sorted_utf8(
        artifact_files
        if (
            isinstance(artifact_files, list)
            and all(
                isinstance(x, str)
                for x in artifact_files
            )
        )
        else []
    )

    # A full-model artifact is any obvious full-model weight
    # artifact rather than the required adapter-only artifact.
    full_model_names = {
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "model.safetensors",
        "model.safetensors.index.json",
    }

    if (
        isinstance(artifact_files, list)
        and any(
            isinstance(x, str)
            and x in full_model_names
            for x in artifact_files
        )
    ):
        codes.append(
            "FULL_MODEL_ARTIFACT"
        )

    # --------------------------------------------------------
    # Base revision + lineage digests
    # --------------------------------------------------------

    base_revision = body.get(
        "baseRevision"
    )

    dataset_digest = body.get(
        "datasetDigest"
    )

    code_digest = body.get(
        "codeDigest"
    )

    config_digest = body.get(
        "configDigest"
    )

    if not adapt_hex(
        base_revision,
        40
    ):
        codes.append(
            "MUTABLE_BASE_REVISION"
        )

    digest_values = (
        dataset_digest,
        code_digest,
        config_digest
    )

    if not all(
        isinstance(value, str)
        and len(value) > 0
        and adapt_hex(value, 64)
        for value in digest_values
    ):
        codes.append(
            "LINEAGE_MISMATCH"
        )

    # expectedDigests binds the supplied lineage evidence.
    expected_digests = body.get(
        "expectedDigests"
    )

    if not isinstance(
        expected_digests,
        dict
    ):
        codes.append(
            "LINEAGE_MISMATCH"
        )
    else:

        expected_dataset = expected_digests.get(
            "datasetDigest"
        )
        expected_code = expected_digests.get(
            "codeDigest"
        )
        expected_config = expected_digests.get(
            "configDigest"
        )

        if (
            expected_dataset is not None
            and expected_dataset != dataset_digest
        ):
            codes.append(
                "LINEAGE_MISMATCH"
            )

        if (
            expected_code is not None
            and expected_code != code_digest
        ):
            codes.append(
                "LINEAGE_MISMATCH"
            )

        if (
            expected_config is not None
            and expected_config != config_digest
        ):
            codes.append(
                "LINEAGE_MISMATCH"
            )

    # --------------------------------------------------------
    # Effective batch
    # --------------------------------------------------------

    micro_batch = body.get(
        "microBatch"
    )

    gradient_accumulation = body.get(
        "gradientAccumulation"
    )

    replicas = body.get(
        "replicas"
    )

    expected_batch = body.get(
        "expectedEffectiveBatch"
    )

    batch_values = (
        micro_batch,
        gradient_accumulation,
        replicas,
        expected_batch
    )

    if not all(
        adapt_safe_integer(x)
        and x > 0
        for x in batch_values
    ):
        codes.append(
            "EFFECTIVE_BATCH_MISMATCH"
        )
    else:

        product = (
            micro_batch
            * gradient_accumulation
            * replicas
        )

        if product != expected_batch:
            codes.append(
                "EFFECTIVE_BATCH_MISMATCH"
            )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint = body.get(
        "checkpoint"
    )

    checkpoint_required = {
        "model",
        "optimizer",
        "scheduler",
        "step",
        "rng",
        "dataPosition"
    }

    if (
        not isinstance(checkpoint, dict)
        or not checkpoint_required.issubset(
            checkpoint.keys()
        )
    ):
        codes.append(
            "INCOMPLETE_CHECKPOINT"
        )

    # --------------------------------------------------------
    # Resume determinism
    # --------------------------------------------------------

    uninterrupted = body.get(
        "uninterruptedWeights"
    )

    resumed = body.get(
        "resumedWeights"
    )

    tolerance = body.get(
        "resumeTolerance"
    )

    resume_arrays_valid = (
        isinstance(uninterrupted, list)
        and isinstance(resumed, list)
        and len(uninterrupted) > 0
        and len(uninterrupted) == len(resumed)
        and all(
            adapt_finite(x)
            for x in uninterrupted
        )
        and all(
            adapt_finite(x)
            for x in resumed
        )
        and adapt_nonnegative_finite(
            tolerance
        )
    )

    if not resume_arrays_valid:

        codes.append(
            "RESUME_DIVERGENCE"
        )

    else:

        for a, b in zip(
            uninterrupted,
            resumed
        ):
            if abs(
                float(a) - float(b)
            ) > float(tolerance):
                codes.append(
                    "RESUME_DIVERGENCE"
                )
                break

    # --------------------------------------------------------
    # Deterministic final response
    # --------------------------------------------------------

    codes = sorted_codes(codes)

    return {
        "labels": labels,
        "templatePass": (
            "CHAT_TEMPLATE_COUNT" not in codes
        ),
        "trainableParams": trainable_names,
        "trainableCount": trainable_count,
        "peftConfigPass": (
            "INVALID_PARAMETER" not in codes
        ),
        "adapterFiles": adapter_files,
        "checkpointComplete": (
            "INCOMPLETE_CHECKPOINT" not in codes
        ),
        "lineagePass": not any(
            code in codes
            for code in (
                "MUTABLE_BASE_REVISION",
                "LINEAGE_MISMATCH"
            )
        ),
        "evalIsolated": not any(
            code in codes
            for code in (
                "EVAL_LEAKAGE",
                "EVAL_DROPOUT_ACTIVE"
            )
        ),
        "evaluationDeterministic": (
            "EVAL_DROPOUT_ACTIVE" not in codes
            and "RESUME_DIVERGENCE" not in codes
        ),
        "resumePass": (
            "RESUME_DIVERGENCE" not in codes
        ),
        "reasonCodes": codes,
    }


# ============================================================
# /adapt
# ============================================================

@app.route("/adapt", methods=["POST"])
def adapt():

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

    operation = body.get("operation")

    if operation not in (
        "choose",
        "repair"
    ):
        return app.response_class(
            response='{"error":"INVALID_INPUT"}',
            status=400,
            mimetype="application/json"
        )

    # --------------------------------------------------------
    # CHOOSE
    # --------------------------------------------------------

    if operation == "choose":

        result = adapt_choose(body)

        if result is None:
            return app.response_class(
                response='{"error":"INVALID_INPUT"}',
                status=400,
                mimetype="application/json"
            )

        return app.response_class(
            response=compact_json(result),
            status=200,
            mimetype="application/json"
        )

    # --------------------------------------------------------
    # REPAIR
    # --------------------------------------------------------

    required_repair = {
        "operation",
        "tokens",
        "templateApplications",
        "parameters",
        "allowedTargets",
        "inferenceMode",
        "trainRowIds",
        "evalRowIds",
        "dropoutActiveDuringEval",
        "artifactFiles",
        "baseRevision",
        "datasetDigest",
        "codeDigest",
        "configDigest",
        "expectedDigests",
        "microBatch",
        "gradientAccumulation",
        "replicas",
        "expectedEffectiveBatch",
        "checkpoint",
        "uninterruptedWeights",
        "resumedWeights",
        "resumeTolerance",
    }

    # The operation itself is valid, but a missing repair
    # field is a repair-level failure rather than the
    # unknown-operation HTTP error.
    if not required_repair.issubset(
        body.keys()
    ):
        return app.response_class(
            response=compact_json({
                "labels": [],
                "templatePass": False,
                "trainableParams": [],
                "trainableCount": 0,
                "peftConfigPass": False,
                "adapterFiles": [],
                "checkpointComplete": False,
                "lineagePass": False,
                "evalIsolated": False,
                "evaluationDeterministic": False,
                "resumePass": False,
                "reasonCodes": [
                    "INVALID_PARAMETER"
                ]
            }),
            status=200,
            mimetype="application/json"
        )

    result = adapt_repair(body)

    return app.response_class(
        response=compact_json(result),
        status=200,
        mimetype="application/json"
    )


# ============================================================
# Q6 - Recover a Content-Addressed ML Pipeline
# Endpoint: POST /pipeline
# ============================================================

PIPELINE_DAG = (
    "verify_data",
    "prepare",
    "train",
    "evaluate",
    "register",
    "publish",
)

PIPELINE_INPUT_NAMES = (
    "generation",
    "checksum",
    "canonicalData",
    "prepareCode",
    "prepareConfig",
    "trainCode",
    "trainConfig",
    "runtime",
    "evaluateCode",
    "evaluateConfig",
    "schemaDigest",
    "publishConfig",
)

PIPELINE_STATUS_VALUES = {
    "started",
    "succeeded",
    "retryable_failed",
    "terminal_failed",
}

PIPELINE_STATES = {}


def pipeline_sha(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def pipeline_compact(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":")
    )


def pipeline_hash_array(values):
    return pipeline_sha(
        pipeline_compact(values)
    )


def pipeline_safe_positive_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= 9007199254740991
    )


def pipeline_valid_nonempty_string(value):
    return (
        isinstance(value, str)
        and value != ""
    )


def pipeline_event_shape(event):
    return (
        isinstance(event, dict)
        and set(event.keys()) == {
            "eventId",
            "revision",
            "node",
            "attempt",
            "status",
            "key",
            "artifactDigest",
            "receiptId",
        }
    )


def pipeline_make_keys(inputs, artifacts):
    """
    Compute the six immutable content-addressed keys.

    A downstream key is null until its parent artifact exists.
    """

    keys = {}

    # verify_data
    keys["verify_data"] = pipeline_hash_array([
        inputs["generation"],
        inputs["checksum"],
    ])

    # prepare
    prepare_artifact = artifacts.get(
        "verify_data"
    )

    if prepare_artifact is None:
        keys["prepare"] = None
    else:
        keys["prepare"] = pipeline_hash_array([
            inputs["canonicalData"],
            inputs["prepareCode"],
            inputs["prepareConfig"],
        ])

    # train
    train_parent = artifacts.get(
        "prepare"
    )

    if train_parent is None:
        keys["train"] = None
    else:
        keys["train"] = pipeline_hash_array([
            train_parent,
            inputs["trainCode"],
            inputs["trainConfig"],
            inputs["runtime"],
        ])

    # evaluate
    evaluate_parent = artifacts.get(
        "train"
    )

    if evaluate_parent is None:
        keys["evaluate"] = None
    else:
        keys["evaluate"] = pipeline_hash_array([
            evaluate_parent,
            inputs["canonicalData"],
            inputs["evaluateCode"],
            inputs["evaluateConfig"],
        ])

    # register
    register_parent = artifacts.get(
        "evaluate"
    )

    if register_parent is None:
        keys["register"] = None
    else:
        keys["register"] = pipeline_hash_array([
            register_parent,
            inputs["schemaDigest"],
        ])

    # publish
    publish_parent = artifacts.get(
        "register"
    )

    if publish_parent is None:
        keys["publish"] = None
    else:
        keys["publish"] = pipeline_hash_array([
            publish_parent,
            inputs["publishConfig"],
        ])

    return keys


def pipeline_dependency_digests(
    node,
    inputs,
    key,
    artifacts
):
    """
    Response dependencyDigests.

    Contains the named inputs relevant to the node plus cacheKey.
    """

    if node == "verify_data":
        result = {
            "generation": inputs["generation"],
            "checksum": inputs["checksum"],
        }

    elif node == "prepare":
        result = {
            "canonicalData": inputs["canonicalData"],
            "prepareCode": inputs["prepareCode"],
            "prepareConfig": inputs["prepareConfig"],
        }

    elif node == "train":
        result = {
            "prepareArtifact": artifacts.get(
                "prepare"
            ),
            "trainCode": inputs["trainCode"],
            "trainConfig": inputs["trainConfig"],
            "runtime": inputs["runtime"],
        }

    elif node == "evaluate":
        result = {
            "trainArtifact": artifacts.get(
                "train"
            ),
            "canonicalData": inputs["canonicalData"],
            "evaluateCode": inputs["evaluateCode"],
            "evaluateConfig": inputs["evaluateConfig"],
        }

    elif node == "register":
        result = {
            "evaluateArtifact": artifacts.get(
                "evaluate"
            ),
            "schemaDigest": inputs["schemaDigest"],
        }

    else:
        result = {
            "registerArtifact": artifacts.get(
                "register"
            ),
            "publishConfig": inputs["publishConfig"],
        }

    result["cacheKey"] = key

    return result


def pipeline_parent(node):
    index = PIPELINE_DAG.index(node)

    if index == 0:
        return None

    return PIPELINE_DAG[index - 1]


def pipeline_descendants_have_terminal(
    node,
    states
):
    parent = pipeline_parent(node)

    while parent is not None:

        state = states.get(parent)

        if (
            state is not None
            and state.get("status")
            == "terminal_failed"
        ):
            return True

        parent = pipeline_parent(parent)

    return False


def pipeline_has_pending_ancestor(
    node,
    states,
    artifacts
):
    parent = pipeline_parent(node)

    while parent is not None:

        if artifacts.get(parent) is not None:
            parent = pipeline_parent(parent)
            continue

        state = states.get(parent)

        if state is None:
            return True

        if state.get("status") in (
            "started",
            "retryable_failed",
        ):
            return True

        parent = pipeline_parent(parent)

    return False


def pipeline_ready(
    node,
    states,
    artifacts,
    keys
):
    if keys.get(node) is None:
        return False

    parent = pipeline_parent(node)

    if parent is None:
        return True

    if artifacts.get(parent) is not None:
        return True

    return False


def pipeline_state_for_key(
    states,
    node,
    key
):
    state = states.get(node)

    if state is None:
        return None

    if state.get("key") != key:
        return None

    return state


def pipeline_event_canonical(event):
    return pipeline_compact(event)


def pipeline_valid_receipt(
    node,
    key,
    receipt
):
    if node in (
        "register",
        "publish",
    ):
        return receipt == (
            f"receipt:{node}:{key}"
        )

    return receipt is None


def pipeline_process_event(
    session_state,
    event,
    current_revision,
    keys,
    artifacts,
    states
):
    """
    Returns:
        "accepted"
        "ignored"
        or raises a conflict code.
    """

    event_id = event["eventId"]

    # --------------------------------------------------------
    # Revision
    # --------------------------------------------------------

    if event["revision"] != current_revision:
        return "ignored"

    node = event["node"]

    if node not in PIPELINE_DAG:
        return "ignored"

    key = event["key"]

    # Wrong/unavailable key.
    if key != keys.get(node):
        return "ignored"

    # Parent must be reusable.
    parent = pipeline_parent(node)

    if parent is not None:

        parent_artifact = artifacts.get(parent)

        if parent_artifact is None:
            return "ignored"

    # --------------------------------------------------------
    # Attempt validation
    # --------------------------------------------------------

    if not pipeline_safe_positive_integer(
        event["attempt"]
    ):
        return "ignored"

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if event["status"] not in PIPELINE_STATUS_VALUES:
        return "ignored"

    # --------------------------------------------------------
    # Artifact validation
    # --------------------------------------------------------

    if event["status"] == "succeeded":

        if not pipeline_valid_nonempty_string(
            event["artifactDigest"]
        ):
            return "ignored"

    else:

        if event["artifactDigest"] is not None:
            return "ignored"

    # --------------------------------------------------------
    # Receipt validation
    # --------------------------------------------------------

    if not pipeline_valid_receipt(
        node,
        key,
        event["receiptId"]
    ):
        return "ignored"

    current = pipeline_state_for_key(
        states,
        node,
        key
    )

    # --------------------------------------------------------
    # No state for this key.
    # --------------------------------------------------------

    if current is None:

        if (
            event["status"] == "started"
            and event["attempt"] == 1
        ):
            states[node] = {
                "key": key,
                "status": "started",
                "attempt": 1,
                "artifactDigest": None,
                "eventId": event_id,
            }

            return "accepted"

        # Completion or attempt > 1 without the
        # initial start is ignored.
        return "ignored"

    previous_status = current["status"]
    previous_attempt = current["attempt"]

    # --------------------------------------------------------
    # Successful/current cache state.
    # --------------------------------------------------------

    if previous_status == "succeeded":

        if event["status"] == "succeeded":

            if (
                event["artifactDigest"]
                != current["artifactDigest"]
            ):
                raise ValueError(
                    "EVIDENCE_CONFLICT"
                )

            return "ignored"

        raise ValueError(
            "STATUS_CONFLICT"
        )

    # --------------------------------------------------------
    # Terminal state.
    # --------------------------------------------------------

    if previous_status == "terminal_failed":
        raise ValueError(
            "STATUS_CONFLICT"
        )

    # --------------------------------------------------------
    # Started state.
    # --------------------------------------------------------

    if previous_status == "started":

        if (
            event["attempt"] < previous_attempt
        ):
            return "ignored"

        if (
            event["attempt"] == previous_attempt
            and event["status"] in (
                "succeeded",
                "retryable_failed",
                "terminal_failed",
            )
        ):
            if event["status"] == "succeeded":

                states[node] = {
                    "key": key,
                    "status": "succeeded",
                    "attempt": previous_attempt,
                    "artifactDigest":
                        event["artifactDigest"],
                    "eventId": event_id,
                }

                # Immutable evidence.
                old_artifact = artifacts.get(
                    node
                )

                if (
                    old_artifact is not None
                    and old_artifact
                    != event["artifactDigest"]
                ):
                    raise ValueError(
                        "EVIDENCE_CONFLICT"
                    )

                artifacts[node] = (
                    event["artifactDigest"]
                )

            elif event["status"] == "retryable_failed":

                states[node] = {
                    "key": key,
                    "status":
                        "retryable_failed",
                    "attempt":
                        previous_attempt,
                    "artifactDigest": None,
                    "eventId": event_id,
                }

            else:

                states[node] = {
                    "key": key,
                    "status":
                        "terminal_failed",
                    "attempt":
                        previous_attempt,
                    "artifactDigest": None,
                    "eventId": event_id,
                }

            return "accepted"

        # started -> started at same attempt is not a valid
        # transition.
        if (
            event["attempt"] == previous_attempt
            and event["status"] == "started"
        ):
            raise ValueError(
                "STATUS_CONFLICT"
            )

        if (
            event["attempt"] > previous_attempt
        ):
            raise ValueError(
                "STATUS_CONFLICT"
            )

        return "ignored"

    # --------------------------------------------------------
    # Retryable failure.
    # --------------------------------------------------------

    if previous_status == "retryable_failed":

        if (
            event["attempt"]
            < previous_attempt
        ):
            return "ignored"

        if (
            event["attempt"]
            == previous_attempt + 1
            and event["status"] == "started"
        ):
            states[node] = {
                "key": key,
                "status": "started",
                "attempt":
                    event["attempt"],
                "artifactDigest": None,
                "eventId": event_id,
            }

            return "accepted"

        if (
            event["attempt"]
            > previous_attempt
        ):
            raise ValueError(
                "STATUS_CONFLICT"
            )

        raise ValueError(
            "STATUS_CONFLICT"
        )

    return "ignored"


def pipeline_build_nodes(
    inputs,
    states,
    artifacts,
    keys
):
    nodes = []

    for node in PIPELINE_DAG:

        key = keys.get(node)

        state = pipeline_state_for_key(
            states,
            node,
            key
        )

        dependency_digests = (
            pipeline_dependency_digests(
                node,
                inputs,
                key,
                artifacts
            )
        )

        triggering = []

        if state is not None:
            triggering = [
                state["eventId"]
            ]

        # ----------------------------------------------------
        # Cached
        # ----------------------------------------------------

        if (
            key is not None
            and artifacts.get(node) is not None
        ):

            nodes.append({
                "node": node,
                "action": "reuse",
                "reasonCodes": [
                    "CACHE_HIT"
                ],
                "dependencyDigests":
                    dependency_digests,
                "triggeringEventIds":
                    triggering,
            })

            continue

        # ----------------------------------------------------
        # Terminal
        # ----------------------------------------------------

        if (
            state is not None
            and state["status"]
            == "terminal_failed"
        ):

            nodes.append({
                "node": node,
                "action": "block",
                "reasonCodes": [
                    "TERMINAL_FAILURE"
                ],
                "dependencyDigests":
                    dependency_digests,
                "triggeringEventIds":
                    triggering,
            })

            continue

        # ----------------------------------------------------
        # Upstream terminal.
        # ----------------------------------------------------

        if pipeline_descendants_have_terminal(
            node,
            states
        ):

            nodes.append({
                "node": node,
                "action": "block",
                "reasonCodes": [
                    "UPSTREAM_TERMINAL"
                ],
                "dependencyDigests":
                    dependency_digests,
                "triggeringEventIds": [],
            })

            continue

        # ----------------------------------------------------
        # Running.
        # ----------------------------------------------------

        if (
            state is not None
            and state["status"] == "started"
        ):

            nodes.append({
                "node": node,
                "action": "block",
                "reasonCodes": [
                    "RUNNING"
                ],
                "dependencyDigests":
                    dependency_digests,
                "triggeringEventIds":
                    triggering,
            })

            continue

        # ----------------------------------------------------
        # Retryable failure.
        # ----------------------------------------------------

        if (
            state is not None
            and state["status"]
            == "retryable_failed"
        ):

            if pipeline_ready(
                node,
                states,
                artifacts,
                keys
            ):

                nodes.append({
                    "node": node,
                    "action": "rerun",
                    "reasonCodes": [
                        "RETRYABLE_FAILURE"
                    ],
                    "dependencyDigests":
                        dependency_digests,
                    "triggeringEventIds":
                        triggering,
                })

            else:

                nodes.append({
                    "node": node,
                    "action": "block",
                    "reasonCodes": [
                        "UPSTREAM_PENDING"
                    ],
                    "dependencyDigests":
                        dependency_digests,
                    "triggeringEventIds": [],
                })

            continue

        # ----------------------------------------------------
        # Ready without cache.
        # ----------------------------------------------------

        if pipeline_ready(
            node,
            states,
            artifacts,
            keys
        ):

            nodes.append({
                "node": node,
                "action": "rerun",
                "reasonCodes": [
                    "CACHE_MISS"
                ],
                "dependencyDigests":
                    dependency_digests,
                "triggeringEventIds": [],
            })

            continue

        # ----------------------------------------------------
        # Pending upstream.
        # ----------------------------------------------------

        nodes.append({
            "node": node,
            "action": "block",
            "reasonCodes": [
                "UPSTREAM_PENDING"
            ],
            "dependencyDigests":
                dependency_digests,
            "triggeringEventIds": [],
        })

    return nodes


@app.route("/pipeline", methods=["POST"])
def pipeline():

    # --------------------------------------------------------
    # JSON request
    # --------------------------------------------------------

    if not request.is_json:
        return app.response_class(
            response='{"error":"INVALID_REQUEST"}',
            status=409,
            mimetype="application/json"
        )

    try:
        body = request.get_json()
    except Exception:
        return app.response_class(
            response='{"error":"INVALID_REQUEST"}',
            status=409,
            mimetype="application/json"
        )

    if not isinstance(body, dict):
        return app.response_class(
            response='{"error":"INVALID_REQUEST"}',
            status=409,
            mimetype="application/json"
        )

    # --------------------------------------------------------
    # Required request fields
    # --------------------------------------------------------

    required = {
        "session",
        "revision",
        "inputs",
        "events",
    }

    if not required.issubset(body.keys()):
        return app.response_class(
            response='{"error":"INVALID_REQUEST"}',
            status=409,
            mimetype="application/json"
        )

    session = body["session"]
    revision = body["revision"]
    inputs = body["inputs"]
    events = body["events"]

    if not pipeline_valid_nonempty_string(
        session
    ):
        return app.response_class(
            response='{"error":"INVALID_REQUEST"}',
            status=409,
            mimetype="application/json"
        )

    if not pipeline_safe_positive_integer(
        revision
    ):
        return app.response_class(
            response='{"error":"INVALID_REQUEST"}',
            status=409,
            mimetype="application/json"
        )

    if not isinstance(inputs, dict):
        return app.response_class(
            response='{"error":"INVALID_REQUEST"}',
            status=409,
            mimetype="application/json"
        )

    if not isinstance(events, list):
        return app.response_class(
            response='{"error":"INVALID_REQUEST"}',
            status=409,
            mimetype="application/json"
        )

    # --------------------------------------------------------
    # Inputs: exactly the required twelve must exist.
    # Extra metadata is allowed.
    # --------------------------------------------------------

    for name in PIPELINE_INPUT_NAMES:

        if (
            name not in inputs
            or not pipeline_valid_nonempty_string(
                inputs[name]
            )
        ):
            return app.response_class(
                response='{"error":"INVALID_REQUEST"}',
                status=409,
                mimetype="application/json"
            )

    # --------------------------------------------------------
    # Session state
    # --------------------------------------------------------

    state = PIPELINE_STATES.get(
        session
    )

    # --------------------------------------------------------
    # New session
    # --------------------------------------------------------

    if state is None:

        state = {
            "revision": revision,
            "input_canonical": pipeline_compact(
                inputs
            ),
            "inputs": dict(inputs),
            "artifacts": {},
            "states": {},
            "events": {},
        }

        PIPELINE_STATES[session] = state

    # --------------------------------------------------------
    # Revision handling
    # --------------------------------------------------------

    elif revision < state["revision"]:

        # Old well-formed events are ignored, but a whole
        # request for an old revision is a request conflict
        # because its inputs are not the current revision.
        if (
            revision != state["revision"]
        ):
            return app.response_class(
                response='{"error":"REVISION_CONFLICT"}',
                status=409,
                mimetype="application/json"
            )

    elif revision == state["revision"]:

        # Same revision must have exactly identical inputs,
        # including extra metadata.
        if (
            pipeline_compact(inputs)
            != state["input_canonical"]
        ):
            return app.response_class(
                response='{"error":"REVISION_CONFLICT"}',
                status=409,
                mimetype="application/json"
            )

    else:
        # New revision.
        #
        # Successful content-addressed cache entries survive,
        # but attempt/terminal state is cleared.
        state["revision"] = revision
        state["input_canonical"] = (
            pipeline_compact(inputs)
        )
        state["inputs"] = dict(inputs)
        state["states"] = {}
        state["events"] = {}

    # --------------------------------------------------------
    # Work on a complete copy so the entire event batch is
    # atomic.
    # --------------------------------------------------------

    import copy

    working_states = copy.deepcopy(
        state["states"]
    )

    working_artifacts = copy.deepcopy(
        state["artifacts"]
    )

    working_events = copy.deepcopy(
        state["events"]
    )

    # --------------------------------------------------------
    # Recompute current keys from reusable artifacts.
    # --------------------------------------------------------

    keys = pipeline_make_keys(
        inputs,
        working_artifacts
    )

    accepted = []
    ignored = []

    # --------------------------------------------------------
    # Validate/process event batch in input order.
    # --------------------------------------------------------

    batch_event_ids = set()

    for event in events:

        # Every event must contain exactly the eight fields.
        if not pipeline_event_shape(event):
            return app.response_class(
                response='{"error":"INVALID_EVENT"}',
                status=409,
                mimetype="application/json"
            )

        event_id = event["eventId"]

        if not pipeline_valid_nonempty_string(
            event_id
        ):
            return app.response_class(
                response='{"error":"INVALID_EVENT"}',
                status=409,
                mimetype="application/json"
            )

        # Canonical event representation.
        canonical_event = (
            pipeline_event_canonical(event)
        )

        # ----------------------------------------------------
        # Existing event ID.
        # ----------------------------------------------------

        if event_id in working_events:

            if (
                working_events[event_id]
                == canonical_event
            ):
                ignored.append(event_id)
                continue

            # Same ID but different event.
            return app.response_class(
                response='{"error":"EVENT_ID_CONFLICT"}',
                status=409,
                mimetype="application/json"
            )

        # Duplicate ID within the same batch.
        if event_id in batch_event_ids:
            return app.response_class(
                response='{"error":"EVENT_ID_CONFLICT"}',
                status=409,
                mimetype="application/json"
            )

        batch_event_ids.add(event_id)

        # ----------------------------------------------------
        # Process event.
        # ----------------------------------------------------

        try:

            result = pipeline_process_event(
                state,
                event,
                revision,
                keys,
                working_artifacts,
                working_states
            )

        except ValueError as exc:

            code = str(exc)

            if code in (
                "EVIDENCE_CONFLICT",
                "STATUS_CONFLICT",
            ):
                return app.response_class(
                    response=compact_json({
                        "error": code
                    }),
                    status=409,
                    mimetype="application/json"
                )

            return app.response_class(
                response='{"error":"INVALID_EVENT"}',
                status=409,
                mimetype="application/json"
            )

        if result == "accepted":

            accepted.append(event_id)

            working_events[event_id] = (
                canonical_event
            )

        else:

            ignored.append(event_id)

    # --------------------------------------------------------
    # Recompute keys after processing successful events.
    # --------------------------------------------------------

    keys = pipeline_make_keys(
        inputs,
        working_artifacts
    )

    # --------------------------------------------------------
    # Commit atomically.
    # --------------------------------------------------------

    state["states"] = working_states
    state["artifacts"] = working_artifacts
    state["events"] = working_events

    # --------------------------------------------------------
    # Build deterministic response.
    # --------------------------------------------------------

    nodes = pipeline_build_nodes(
        inputs,
        working_states,
        working_artifacts,
        keys
    )

    return app.response_class(
        response=compact_json({
            "revision": revision,
            "acceptedEventIds": accepted,
            "ignoredEventIds": ignored,
            "nodes": nodes,
        }),
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
    port = int(os.environ.get("PORT", 8080))

    app.run(
        host="0.0.0.0",
        port=port
    )
    app.run(host="0.0.0.0", port=5000)