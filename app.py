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
# Q5 - Quantize and Admit a Model Under Explicit Constraints
# Endpoint: POST /quantize
# ============================================================

QUANTIZE_FREEZES = {}
QUANTIZE_SAFE_MAX = 2**53 - 1


def quantize_safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= QUANTIZE_SAFE_MAX
    )


def quantize_positive_safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= QUANTIZE_SAFE_MAX
    )


def quantize_finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def quantize_nonnegative_finite(value):
    return (
        quantize_finite(value)
        and float(value) >= 0
    )


def quantize_unit(value):
    return (
        quantize_finite(value)
        and 0 <= float(value) <= 1
    )


def quantize_digest(value):
    return (
        isinstance(value, str)
        and value != ""
    )


def quantize_sha256(value):
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def quantize_unique_nonempty_strings(value):
    if not isinstance(value, list):
        return False

    if len(value) == 0:
        return False

    if any(
        not isinstance(x, str) or x == ""
        for x in value
    ):
        return False

    return len(set(value)) == len(value)


# ------------------------------------------------------------
# Inventory calculation / validation
# ------------------------------------------------------------

def quantize_inventory(inventory):
    """
    Recompute the inventory, total bytes and package digest.

    Inventory entries use the exact key order:
        name, bytes, sha256
    """

    if not isinstance(inventory, list):
        return None, None, None, False

    seen = set()
    normalized = []

    for item in inventory:

        if not isinstance(item, dict):
            return None, None, None, False

        if set(item.keys()) != {
            "name",
            "bytes",
            "sha256"
        }:
            return None, None, None, False

        name = item["name"]
        byte_count = item["bytes"]
        digest = item["sha256"]

        if (
            not isinstance(name, str)
            or name == ""
            or name in seen
        ):
            return None, None, None, False

        if not quantize_safe_integer(byte_count):
            return None, None, None, False

        if not quantize_sha256(digest):
            return None, None, None, False

        seen.add(name)

        normalized.append({
            "name": name,
            "bytes": byte_count,
            "sha256": digest
        })

    normalized.sort(
        key=lambda x: utf8(x["name"])
    )

    total = 0

    for item in normalized:
        total += item["bytes"]

        if total > QUANTIZE_SAFE_MAX:
            return None, None, None, False

    package_digest = hashlib.sha256(
        compact_json(normalized).encode("utf-8")
    ).hexdigest()

    return (
        normalized,
        total,
        package_digest,
        True
    )


# ------------------------------------------------------------
# Freeze candidate
# ------------------------------------------------------------

def quantize_freeze_candidate(
    candidate,
    calibration_digest,
    tokenizer_digest,
    allowed_reasons
):

    if not isinstance(candidate, dict):
        return {
            "name": "",
            "status": "invalid",
            "inventory": [],
            "totalBytes": None,
            "packageDigest": None,
            "reasonCodes": [
                "INVALID_INPUT"
            ]
        }

    required = {
        "name",
        "files",
        "loadable",
        "calibrationDigest",
        "tokenizerDigest"
    }

    optional = {
        "unsupportedReason"
    }

    if (
        not required.issubset(candidate.keys())
        or not set(candidate.keys()).issubset(
            required | optional
        )
    ):
        return {
            "name": (
                candidate.get("name", "")
                if isinstance(candidate.get("name"), str)
                else ""
            ),
            "status": "invalid",
            "inventory": [],
            "totalBytes": None,
            "packageDigest": None,
            "reasonCodes": [
                "INVALID_INPUT"
            ]
        }

    name = candidate["name"]
    files = candidate["files"]
    loadable = candidate["loadable"]
    candidate_calibration = candidate["calibrationDigest"]
    candidate_tokenizer = candidate["tokenizerDigest"]
    unsupported_reason = candidate.get("unsupportedReason")

    codes = []

    # Basic candidate validation
    if not isinstance(name, str) or name == "":
        codes.append("INVALID_INPUT")

    if not isinstance(files, dict):
        codes.append("INVALID_INPUT")
        files = {}
    else:
        for filename, content in files.items():
            if (
                not isinstance(filename, str)
                or filename == ""
                or not isinstance(content, str)
            ):
                codes.append("INVALID_INPUT")

    if not isinstance(loadable, bool):
        codes.append("INVALID_INPUT")

    if not quantize_digest(candidate_calibration):
        codes.append("INVALID_INPUT")

    if not quantize_digest(candidate_tokenizer):
        codes.append("INVALID_INPUT")

    if unsupported_reason is not None:
        if (
            not isinstance(unsupported_reason, str)
            or unsupported_reason == ""
        ):
            codes.append("INVALID_INPUT")

    if codes:
        return {
            "name": name if isinstance(name, str) else "",
            "status": "invalid",
            "inventory": [],
            "totalBytes": None,
            "packageDigest": None,
            "reasonCodes": sorted_codes(codes)
        }

    # --------------------------------------------------------
    # Build exact inventory from UTF-8 file contents
    # --------------------------------------------------------

    inventory = []

    for filename, content in files.items():

        raw = content.encode("utf-8")

        inventory.append({
            "name": filename,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()
        })

    inventory.sort(
        key=lambda x: utf8(x["name"])
    )

    total_bytes = sum(
        item["bytes"]
        for item in inventory
    )

    if total_bytes > QUANTIZE_SAFE_MAX:
        return {
            "name": name,
            "status": "invalid",
            "inventory": [],
            "totalBytes": None,
            "packageDigest": None,
            "reasonCodes": [
                "INVALID_INPUT"
            ]
        }

    package_digest = hashlib.sha256(
        compact_json(inventory).encode("utf-8")
    ).hexdigest()

    # --------------------------------------------------------
    # Unsupported candidate
    # --------------------------------------------------------

    if unsupported_reason is not None:

        if unsupported_reason in allowed_reasons:

            return {
                "name": name,
                "status": "unsupported",
                "inventory": inventory,
                "totalBytes": total_bytes,
                "packageDigest": package_digest,
                "reasonCodes": []
            }

        return {
            "name": name,
            "status": "invalid",
            "inventory": inventory,
            "totalBytes": total_bytes,
            "packageDigest": package_digest,
            "reasonCodes": [
                "UNALLOWED_UNSUPPORTED_REASON"
            ]
        }

    # --------------------------------------------------------
    # Normal frozen candidate
    # --------------------------------------------------------

    if not loadable:
        codes.append("NOT_LOADABLE")

    if candidate_calibration != calibration_digest:
        codes.append("CALIBRATION_MISMATCH")

    if candidate_tokenizer != tokenizer_digest:
        codes.append("TOKENIZER_MISMATCH")

    status = (
        "invalid"
        if codes
        else "frozen"
    )

    return {
        "name": name,
        "status": status,
        "inventory": (
            inventory
            if status != "invalid"
            else []
        ),
        "totalBytes": (
            total_bytes
            if status != "invalid"
            else None
        ),
        "packageDigest": (
            package_digest
            if status != "invalid"
            else None
        ),
        "reasonCodes": sorted_codes(codes)
    }


# ------------------------------------------------------------
# Freeze request
# ------------------------------------------------------------

def quantize_freeze(body):

    required = {
        "phase",
        "freezeId",
        "calibrationDigest",
        "tokenizerDigest",
        "allowedUnsupportedReasons",
        "candidates"
    }

    if (
        not isinstance(body, dict)
        or not required.issubset(body.keys())
        or body.get("phase") != "freeze"
    ):
        return None

    freeze_id = body["freezeId"]

    if (
        not isinstance(freeze_id, str)
        or freeze_id == ""
        or len(freeze_id) > 128
    ):
        return None

    calibration_digest = body["calibrationDigest"]
    tokenizer_digest = body["tokenizerDigest"]

    if not quantize_digest(calibration_digest):
        return None

    if not quantize_digest(tokenizer_digest):
        return None

    allowed_reasons = body["allowedUnsupportedReasons"]

    if not quantize_unique_nonempty_strings(
        allowed_reasons
    ):
        return None

    candidates = body["candidates"]

    if (
        not isinstance(candidates, list)
        or len(candidates) == 0
    ):
        return None

    names = []

    for candidate in candidates:

        if not isinstance(candidate, dict):
            return None

        name = candidate.get("name")

        if (
            not isinstance(name, str)
            or name == ""
        ):
            return None

        names.append(name)

    if len(set(names)) != len(names):
        return None

    results = []

    for candidate in candidates:

        results.append(
            quantize_freeze_candidate(
                candidate,
                calibration_digest,
                tokenizer_digest,
                set(allowed_reasons)
            )
        )

    results.sort(
        key=lambda x: utf8(x["name"])
    )

    return {
        "freezeId": freeze_id,
        "candidates": results
    }


# ------------------------------------------------------------
# Validate stored/submitted frozen candidate
# ------------------------------------------------------------

def quantize_validate_frozen_candidate(candidate):

    if not isinstance(candidate, dict):
        return False

    if set(candidate.keys()) != {
        "name",
        "status",
        "inventory",
        "totalBytes",
        "packageDigest",
        "reasonCodes"
    }:
        return False

    if (
        not isinstance(candidate["name"], str)
        or candidate["name"] == ""
    ):
        return False

    if candidate["status"] not in (
        "frozen",
        "unsupported",
        "invalid"
    ):
        return False

    if not isinstance(
        candidate["reasonCodes"],
        list
    ):
        return False

    if any(
        not isinstance(x, str)
        for x in candidate["reasonCodes"]
    ):
        return False

    inventory_result = quantize_inventory(
        candidate["inventory"]
    )

    inventory, total, digest, valid = inventory_result

    if not valid:
        return False

    if candidate["status"] == "invalid":

        if (
            candidate["inventory"] != []
            or candidate["totalBytes"] is not None
            or candidate["packageDigest"] is not None
        ):
            return False

    else:

        if candidate["totalBytes"] != total:
            return False

        if candidate["packageDigest"] != digest:
            return False

    return True


# ------------------------------------------------------------
# Prediction validation / accuracy
# ------------------------------------------------------------

def quantize_accuracy(rows, candidate_name):

    if not isinstance(rows, list):
        return None, {}, False

    if len(rows) == 0:
        return None, {}, False

    correct = 0
    slice_counts = {}

    for row in rows:

        if not isinstance(row, dict):
            return None, {}, False

        if set(row.keys()) != {
            "label",
            "slice",
            "predictions"
        }:
            return None, {}, False

        label = row["label"]
        slice_name = row["slice"]
        predictions = row["predictions"]

        if (
            isinstance(label, bool)
            or not isinstance(label, int)
            or label not in (0, 1)
        ):
            return None, {}, False

        if (
            not isinstance(slice_name, str)
            or slice_name == ""
        ):
            return None, {}, False

        if not isinstance(predictions, dict):
            return None, {}, False

        if candidate_name not in predictions:
            return None, {}, False

        prediction = predictions[candidate_name]

        if (
            isinstance(prediction, bool)
            or not isinstance(prediction, int)
            or prediction not in (0, 1)
        ):
            return None, {}, False

        is_correct = prediction == label

        if is_correct:
            correct += 1

        if slice_name not in slice_counts:
            slice_counts[slice_name] = [0, 0]

        slice_counts[slice_name][1] += 1

        if is_correct:
            slice_counts[slice_name][0] += 1

    aggregate = round(
        correct / len(rows),
        12
    )

    slices = {}

    for name, counts in slice_counts.items():

        slices[name] = round(
            counts[0] / counts[1],
            12
        )

    return aggregate, slices, True


# ------------------------------------------------------------
# Select candidate
# ------------------------------------------------------------

def quantize_select(body):

    required = {
        "phase",
        "freezeId",
        "candidates",
        "policy",
        "latencies",
        "rows"
    }

    if (
        not isinstance(body, dict)
        or not required.issubset(body.keys())
        or body.get("phase") != "select"
    ):
        return None

    freeze_id = body["freezeId"]

    if (
        not isinstance(freeze_id, str)
        or freeze_id == ""
    ):
        return None

    candidates = body["candidates"]
    policy = body["policy"]
    latencies = body["latencies"]
    rows = body["rows"]

    if not isinstance(candidates, list):
        return None

    if not isinstance(policy, dict):
        return None

    if not isinstance(latencies, dict):
        return None

    if not isinstance(rows, list):
        return None

    global_codes = []

    # --------------------------------------------------------
    # Find freeze
    # --------------------------------------------------------

    frozen = QUANTIZE_FREEZES.get(freeze_id)

    if frozen is None:
        global_codes.append("NOT_FROZEN")

    # --------------------------------------------------------
    # Validate policy
    # --------------------------------------------------------

    policy_required = {
        "maxBytes",
        "aggregateFloor",
        "requiredSlices",
        "maxLatencyMs",
        "candidateOrder"
    }

    policy_valid = (
        set(policy.keys()) == policy_required
    )

    if not policy_valid:

        global_codes.append("INVALID_POLICY")

    else:

        if not quantize_safe_integer(
            policy["maxBytes"]
        ):
            global_codes.append("INVALID_POLICY")

        if not quantize_unit(
            policy["aggregateFloor"]
        ):
            global_codes.append("INVALID_POLICY")

        required_slices = policy["requiredSlices"]

        if not isinstance(
            required_slices,
            dict
        ):
            global_codes.append("INVALID_POLICY")

        else:

            seen_slice_names = set()

            for name, floor in required_slices.items():

                if (
                    not isinstance(name, str)
                    or name == ""
                    or name in seen_slice_names
                    or not quantize_unit(floor)
                ):
                    global_codes.append("INVALID_POLICY")

                seen_slice_names.add(name)

        if not quantize_nonnegative_finite(
            policy["maxLatencyMs"]
        ):
            global_codes.append("INVALID_POLICY")

        if not quantize_unique_nonempty_strings(
            policy["candidateOrder"]
        ):
            global_codes.append("INVALID_POLICY")

    # --------------------------------------------------------
    # Frozen candidate array must exactly match stored response
    # --------------------------------------------------------

    supplied_valid = True

    if frozen is not None:

        # IMPORTANT:
        # QUANTIZE_FREEZES stores:
        #
        # {
        #     "fingerprint": ...,
        #     "response": {
        #         "freezeId": ...,
        #         "candidates": [...]
        #     },
        #     "response_json": ...
        # }
        #
        # Therefore candidates are under frozen["response"].

        stored_candidates = frozen["response"]["candidates"]

        if candidates != stored_candidates:

            global_codes.append(
                "INVALID_LINEAGE"
            )

            supplied_valid = False

        else:

            for candidate in candidates:

                if not quantize_validate_frozen_candidate(
                    candidate
                ):
                    global_codes.append(
                        "INVALID_MANIFEST"
                    )

                    supplied_valid = False

    # --------------------------------------------------------
    # Candidate names must equal candidateOrder
    # --------------------------------------------------------

    if policy_valid:

        order = policy["candidateOrder"]

        supplied_names = []

        for candidate in candidates:

            if isinstance(candidate, dict):
                name = candidate.get("name")

                if isinstance(name, str):
                    supplied_names.append(name)

        if (
            len(supplied_names) != len(set(supplied_names))
            or set(supplied_names) != set(order)
        ):
            global_codes.append(
                "INVALID_POLICY"
            )

    else:

        order = []

    # --------------------------------------------------------
    # Validate latency map
    # --------------------------------------------------------

    for name, value in latencies.items():

        if not isinstance(name, str):
            global_codes.append("INVALID_POLICY")
            continue

        if not quantize_nonnegative_finite(value):
            global_codes.append("INVALID_POLICY")

    # --------------------------------------------------------
    # Result ordering
    # --------------------------------------------------------

    order_index = {
        name: index
        for index, name in enumerate(order)
    }

    results_by_name = {}

    # --------------------------------------------------------
    # Evaluate candidates
    # --------------------------------------------------------

    for candidate in candidates:

        if not isinstance(candidate, dict):
            continue

        name = candidate.get("name")

        if not isinstance(name, str):
            continue

        codes = []

        aggregate = None
        slices = {}

        total_bytes = None
        latency = None

        # ----------------------------------------------------
        # Manifest
        # ----------------------------------------------------

        manifest_valid = (
            quantize_validate_frozen_candidate(
                candidate
            )
        )

        if not manifest_valid:

            codes.append("INVALID_MANIFEST")

        else:

            inventory_result = quantize_inventory(
                candidate["inventory"]
            )

            inventory, recomputed_bytes, recomputed_digest, valid = (
                inventory_result
            )

            if not valid:

                codes.append("INVALID_MANIFEST")

            else:

                # Never trust submitted totalBytes.
                total_bytes = recomputed_bytes

                if (
                    candidate["totalBytes"]
                    != recomputed_bytes
                    or candidate["packageDigest"]
                    != recomputed_digest
                ):
                    codes.append(
                        "INVALID_MANIFEST"
                    )

        # ----------------------------------------------------
        # Only frozen candidates may be admitted
        # ----------------------------------------------------

        if candidate.get("status") != "frozen":

            codes.append(
                "INVALID_LINEAGE"
            )

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        if name not in latencies:

            codes.append(
                "INVALID_POLICY"
            )

        else:

            latency_value = latencies[name]

            if quantize_nonnegative_finite(
                latency_value
            ):
                latency = latency_value

            else:
                codes.append(
                    "INVALID_POLICY"
                )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        if len(rows) == 0:

            codes.append(
                "INVALID_PREDICTIONS"
            )

            # Required slices have no validated values.
            if policy_valid:

                slices = {
                    slice_name: None
                    for slice_name in
                    policy["requiredSlices"].keys()
                }

        else:

            aggregate, calculated_slices, prediction_valid = (
                quantize_accuracy(
                    rows,
                    name
                )
            )

            if not prediction_valid:

                aggregate = None

                if policy_valid:

                    slices = {
                        slice_name: None
                        for slice_name in
                        policy["requiredSlices"].keys()
                    }

                else:

                    slices = {}

                codes.append(
                    "INVALID_PREDICTIONS"
                )

            else:

                slices = calculated_slices

        # ----------------------------------------------------
        # Aggregate floor
        # ----------------------------------------------------

        if (
            aggregate is not None
            and policy_valid
            and aggregate < policy["aggregateFloor"]
        ):

            codes.append(
                "AGGREGATE_FLOOR"
            )

        # ----------------------------------------------------
        # Required slices
        # ----------------------------------------------------

        if (
            policy_valid
            and aggregate is not None
        ):

            for slice_name, floor in (
                policy["requiredSlices"].items()
            ):

                if slice_name not in slices:

                    codes.append(
                        f"MISSING_SLICE:{slice_name}"
                    )

                elif slices[slice_name] < floor:

                    codes.append(
                        f"SLICE_FLOOR:{slice_name}"
                    )

        # ----------------------------------------------------
        # Size limit
        # ----------------------------------------------------

        if (
            total_bytes is not None
            and policy_valid
            and total_bytes > policy["maxBytes"]
        ):

            codes.append(
                "SIZE_LIMIT"
            )

        # ----------------------------------------------------
        # Latency limit
        # ----------------------------------------------------

        if (
            latency is not None
            and policy_valid
            and latency > policy["maxLatencyMs"]
        ):

            codes.append(
                "LATENCY_LIMIT"
            )

        codes = sorted_codes(codes)

        results_by_name[name] = {
            "name": name,
            "aggregate": aggregate,
            "slices": slices,
            "totalBytes": total_bytes,
            "latencyMs": latency,
            "admitted": len(codes) == 0,
            "reasonCodes": codes
        }

    # --------------------------------------------------------
    # Results ordered by candidateOrder
    # --------------------------------------------------------

    results = sorted(
        results_by_name.values(),
        key=lambda result: (
            order_index.get(
                result["name"],
                len(order)
            ),
            utf8(result["name"])
        )
    )

    # --------------------------------------------------------
    # Select winner
    # --------------------------------------------------------

    admitted = [
        result
        for result in results
        if result["admitted"]
    ]

    if admitted and not global_codes:

        winner = min(
            admitted,
            key=lambda result: (
                result["totalBytes"],
                result["latencyMs"],
                order_index.get(
                    result["name"],
                    len(order)
                )
            )
        )

        selected = winner["name"]

        # packageManifest must be the recorded winner object.
        package_manifest = winner.copy()

    else:

        selected = None
        package_manifest = None

    return {
        "freezeId": freeze_id,
        "selected": selected,
        "results": results,
        "packageManifest": package_manifest
    }


# ============================================================
# /quantize
# ============================================================

@app.route("/quantize", methods=["POST"])
def quantize():

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

    if phase not in (
        "freeze",
        "select"
    ):

        return app.response_class(
            response='{"error":"INVALID_INPUT"}',
            status=400,
            mimetype="application/json"
        )

    # --------------------------------------------------------
    # FREEZE
    # --------------------------------------------------------

    if phase == "freeze":

        result = quantize_freeze(body)

        if result is None:

            return app.response_class(
                response='{"error":"INVALID_INPUT"}',
                status=400,
                mimetype="application/json"
            )

        freeze_id = result["freezeId"]

        # Fingerprint entire freeze request.
        fingerprint = hashlib.sha256(
            json.dumps(
                body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

        existing = QUANTIZE_FREEZES.get(
            freeze_id
        )

        # ----------------------------------------------------
        # Identical replay
        # ----------------------------------------------------

        if existing is not None:

            if existing["fingerprint"] == fingerprint:

                return app.response_class(
                    response=existing["response_json"],
                    status=200,
                    mimetype="application/json"
                )

            # Same freezeId but different input.
            return app.response_class(
                response='{"error":"FREEZE_ID_CONFLICT"}',
                status=409,
                mimetype="application/json"
            )

        response_json = compact_json(result)

        # Persist complete response.
        QUANTIZE_FREEZES[freeze_id] = {
            "fingerprint": fingerprint,
            "response": result,
            "response_json": response_json
        }

        return app.response_class(
            response=response_json,
            status=200,
            mimetype="application/json"
        )

    # --------------------------------------------------------
    # SELECT
    # --------------------------------------------------------

    result = quantize_select(body)

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


# ============================================================
# Q6 - Recover a Content-Addressed ML Pipeline
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

PIPELINE_STATUSES = {
    "started",
    "succeeded",
    "retryable_failed",
    "terminal_failed",
}

# State is isolated by session.
PIPELINE_SESSIONS = {}

PIPELINE_LOCK = threading.RLock()


# ------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------

def pipeline_nonempty_string(value):
    return isinstance(value, str) and len(value) > 0


def pipeline_safe_positive_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= (2 ** 53 - 1)
    )


def pipeline_canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def pipeline_hash_array(values):
    raw = pipeline_canonical_json(values).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def pipeline_event_json(event):
    return pipeline_canonical_json(event)


def pipeline_utf8_key(value):
    return value.encode("utf-8")


def pipeline_parent(node):
    index = PIPELINE_DAG.index(node)

    if index == 0:
        return None

    return PIPELINE_DAG[index - 1]


# ------------------------------------------------------------
# Cache helpers
# ------------------------------------------------------------

def pipeline_cache_entry_reusable(cache, node, key):
    """
    A cache entry is reusable ONLY when:

    1. a current key exists,
    2. a cache entry exists,
    3. the stored cache key exactly equals the current key,
    4. an immutable artifact digest exists,
    5. an immutable success event ID exists.
    """

    if key is None:
        return False

    entry = cache.get(node)

    if not isinstance(entry, dict):
        return False

    if entry.get("key") != key:
        return False

    if not pipeline_nonempty_string(
        entry.get("artifactDigest")
    ):
        return False

    if not pipeline_nonempty_string(
        entry.get("eventId")
    ):
        return False

    return True


def pipeline_cache_artifact(cache, node, key):
    if not pipeline_cache_entry_reusable(
        cache,
        node,
        key,
    ):
        return None

    return cache[node]["artifactDigest"]


# ------------------------------------------------------------
# Content-addressed keys
# ------------------------------------------------------------

def pipeline_compute_keys(inputs, cache):
    """
    Compute the fixed DAG keys.

    IMPORTANT:
    A child key is NULL until its parent is reusable
    under the CURRENT parent key.
    """

    keys = {}

    # --------------------------------------------------------
    # verify_data
    # [generation, checksum]
    # --------------------------------------------------------

    keys["verify_data"] = pipeline_hash_array([
        inputs["generation"],
        inputs["checksum"],
    ])

    # --------------------------------------------------------
    # prepare
    # [canonicalData, prepareCode, prepareConfig]
    # --------------------------------------------------------

    if pipeline_cache_entry_reusable(
        cache,
        "verify_data",
        keys["verify_data"],
    ):
        keys["prepare"] = pipeline_hash_array([
            inputs["canonicalData"],
            inputs["prepareCode"],
            inputs["prepareConfig"],
        ])
    else:
        keys["prepare"] = None

    # --------------------------------------------------------
    # train
    # [prepareArtifact, trainCode, trainConfig, runtime]
    # --------------------------------------------------------

    prepare_artifact = pipeline_cache_artifact(
        cache,
        "prepare",
        keys["prepare"],
    )

    if prepare_artifact is not None:
        keys["train"] = pipeline_hash_array([
            prepare_artifact,
            inputs["trainCode"],
            inputs["trainConfig"],
            inputs["runtime"],
        ])
    else:
        keys["train"] = None

    # --------------------------------------------------------
    # evaluate
    # [trainArtifact, canonicalData,
    #  evaluateCode, evaluateConfig]
    # --------------------------------------------------------

    train_artifact = pipeline_cache_artifact(
        cache,
        "train",
        keys["train"],
    )

    if train_artifact is not None:
        keys["evaluate"] = pipeline_hash_array([
            train_artifact,
            inputs["canonicalData"],
            inputs["evaluateCode"],
            inputs["evaluateConfig"],
        ])
    else:
        keys["evaluate"] = None

    # --------------------------------------------------------
    # register
    # [evaluateArtifact, schemaDigest]
    # --------------------------------------------------------

    evaluate_artifact = pipeline_cache_artifact(
        cache,
        "evaluate",
        keys["evaluate"],
    )

    if evaluate_artifact is not None:
        keys["register"] = pipeline_hash_array([
            evaluate_artifact,
            inputs["schemaDigest"],
        ])
    else:
        keys["register"] = None

    # --------------------------------------------------------
    # publish
    # [registerArtifact, publishConfig]
    # --------------------------------------------------------

    register_artifact = pipeline_cache_artifact(
        cache,
        "register",
        keys["register"],
    )

    if register_artifact is not None:
        keys["publish"] = pipeline_hash_array([
            register_artifact,
            inputs["publishConfig"],
        ])
    else:
        keys["publish"] = None

    return keys


# ------------------------------------------------------------
# Dependency output
# ------------------------------------------------------------

def pipeline_dependencies(
    node,
    inputs,
    cache,
    keys,
):
    """
    dependencyDigests must contain the named inputs for the node
    followed by cacheKey.

    Parent artifact values are exposed ONLY when the parent is
    reusable for the current key.
    """

    if node == "verify_data":

        deps = {
            "generation": inputs["generation"],
            "checksum": inputs["checksum"],
        }

    elif node == "prepare":

        deps = {
            "canonicalData": inputs["canonicalData"],
            "prepareCode": inputs["prepareCode"],
            "prepareConfig": inputs["prepareConfig"],
        }

    elif node == "train":

        deps = {
            "prepareArtifact": pipeline_cache_artifact(
                cache,
                "prepare",
                keys["prepare"],
            ),
            "trainCode": inputs["trainCode"],
            "trainConfig": inputs["trainConfig"],
            "runtime": inputs["runtime"],
        }

    elif node == "evaluate":

        deps = {
            "trainArtifact": pipeline_cache_artifact(
                cache,
                "train",
                keys["train"],
            ),
            "canonicalData": inputs["canonicalData"],
            "evaluateCode": inputs["evaluateCode"],
            "evaluateConfig": inputs["evaluateConfig"],
        }

    elif node == "register":

        deps = {
            "evaluateArtifact": pipeline_cache_artifact(
                cache,
                "evaluate",
                keys["evaluate"],
            ),
            "schemaDigest": inputs["schemaDigest"],
        }

    else:

        deps = {
            "registerArtifact": pipeline_cache_artifact(
                cache,
                "register",
                keys["register"],
            ),
            "publishConfig": inputs["publishConfig"],
        }

    deps["cacheKey"] = keys[node]

    return deps


# ------------------------------------------------------------
# Ancestor status helpers
# ------------------------------------------------------------

def pipeline_find_upstream_block(
    node,
    states,
    cache,
    keys,
):
    """
    Determine whether an upstream node is terminal or pending.

    Returns:
        "TERMINAL"
        "PENDING"
        None
    """

    index = PIPELINE_DAG.index(node)

    if index == 0:
        return None

    for i in range(index):

        ancestor = PIPELINE_DAG[i]
        ancestor_key = keys[ancestor]

        # If the ancestor cannot currently produce a key,
        # inspect its state.
        if ancestor_key is None:

            state = states.get(ancestor)

            if (
                state is not None
                and state.get("status") == "terminal_failed"
            ):
                return "TERMINAL"

            return "PENDING"

        # If ancestor isn't reusable, inspect its current state.
        if not pipeline_cache_entry_reusable(
            cache,
            ancestor,
            ancestor_key,
        ):

            state = states.get(ancestor)

            if (
                state is not None
                and state.get("key") == ancestor_key
                and state.get("status") == "terminal_failed"
            ):
                return "TERMINAL"

            return "PENDING"

    return None


# ------------------------------------------------------------
# Build response nodes
# ------------------------------------------------------------

def pipeline_build_nodes(
    inputs,
    cache,
    states,
    keys,
):
    nodes = []

    for node in PIPELINE_DAG:

        key = keys[node]

        dependency_digests = pipeline_dependencies(
            node,
            inputs,
            cache,
            keys,
        )

        # ----------------------------------------------------
        # CACHE HIT
        # ----------------------------------------------------

        if pipeline_cache_entry_reusable(
            cache,
            node,
            key,
        ):

            nodes.append({
                "node": node,
                "action": "reuse",
                "reasonCodes": [
                    "CACHE_HIT"
                ],
                "dependencyDigests": dependency_digests,
                "triggeringEventIds": [
                    cache[node]["eventId"]
                ],
            })

            continue

        # ----------------------------------------------------
        # If this node itself is blocked by an upstream node,
        # determine whether that upstream node is terminal
        # or merely pending.
        # ----------------------------------------------------

        upstream_status = pipeline_find_upstream_block(
            node,
            states,
            cache,
            keys,
        )

        if upstream_status == "TERMINAL":

            nodes.append({
                "node": node,
                "action": "block",
                "reasonCodes": [
                    "UPSTREAM_TERMINAL"
                ],
                "dependencyDigests": dependency_digests,
                "triggeringEventIds": [],
            })

            continue

        if upstream_status == "PENDING":

            nodes.append({
                "node": node,
                "action": "block",
                "reasonCodes": [
                    "UPSTREAM_PENDING"
                ],
                "dependencyDigests": dependency_digests,
                "triggeringEventIds": [],
            })

            continue

        # ----------------------------------------------------
        # Current node state
        # ----------------------------------------------------

        state = states.get(node)

        if (
            state is not None
            and state.get("key") == key
        ):

            status = state.get("status")

            if status == "started":

                nodes.append({
                    "node": node,
                    "action": "block",
                    "reasonCodes": [
                        "RUNNING"
                    ],
                    "dependencyDigests": dependency_digests,
                    "triggeringEventIds": [
                        state["eventId"]
                    ],
                })

                continue

            if status == "terminal_failed":

                nodes.append({
                    "node": node,
                    "action": "block",
                    "reasonCodes": [
                        "TERMINAL_FAILURE"
                    ],
                    "dependencyDigests": dependency_digests,
                    "triggeringEventIds": [
                        state["eventId"]
                    ],
                })

                continue

            if status == "retryable_failed":

                nodes.append({
                    "node": node,
                    "action": "rerun",
                    "reasonCodes": [
                        "RETRYABLE_FAILURE"
                    ],
                    "dependencyDigests": dependency_digests,
                    "triggeringEventIds": [
                        state["eventId"]
                    ],
                })

                continue

        # ----------------------------------------------------
        # No cache and no blocking state
        # ----------------------------------------------------

        if key is None:

            nodes.append({
                "node": node,
                "action": "block",
                "reasonCodes": [
                    "UPSTREAM_PENDING"
                ],
                "dependencyDigests": dependency_digests,
                "triggeringEventIds": [],
            })

        else:

            nodes.append({
                "node": node,
                "action": "rerun",
                "reasonCodes": [
                    "CACHE_MISS"
                ],
                "dependencyDigests": dependency_digests,
                "triggeringEventIds": [],
            })

    return nodes


# ------------------------------------------------------------
# Event validation
# ------------------------------------------------------------

def pipeline_valid_event(event):

    if not isinstance(event, dict):
        return False

    # EXACTLY eight fields
    if set(event.keys()) != {
        "eventId",
        "revision",
        "node",
        "attempt",
        "status",
        "key",
        "artifactDigest",
        "receiptId",
    }:
        return False

    # eventId
    if not pipeline_nonempty_string(
        event["eventId"]
    ):
        return False

    # revision
    if not pipeline_safe_positive_int(
        event["revision"]
    ):
        return False

    # node
    if event["node"] not in PIPELINE_DAG:
        return False

    # attempt
    if not pipeline_safe_positive_int(
        event["attempt"]
    ):
        return False

    # status
    if event["status"] not in PIPELINE_STATUSES:
        return False

    # key must be a non-empty string
    if not pipeline_nonempty_string(
        event["key"]
    ):
        return False

    status = event["status"]
    node = event["node"]

    # --------------------------------------------------------
    # Artifact rules
    # --------------------------------------------------------

    if status == "succeeded":

        if not pipeline_nonempty_string(
            event["artifactDigest"]
        ):
            return False

    else:

        if event["artifactDigest"] is not None:
            return False

    # --------------------------------------------------------
    # Receipt rules
    # --------------------------------------------------------

    if status == "succeeded" and node in (
        "register",
        "publish",
    ):

        expected = (
            "receipt:"
            + node
            + ":"
            + event["key"]
        )

        if event["receiptId"] != expected:
            return False

    else:

        if event["receiptId"] is not None:
            return False

    return True


# ------------------------------------------------------------
# Apply one event
# ------------------------------------------------------------

def pipeline_apply_event(
    event,
    inputs,
    cache,
    states,
):
    """
    Return one of:

        "accepted"
        "ignored"
        "status_conflict"
        "evidence_conflict"

    No mutation occurs for ignored/conflicting transitions
    except accepted events.
    """

    node = event["node"]

    # Recompute keys using the CURRENT cache.
    keys = pipeline_compute_keys(
        inputs,
        cache,
    )

    current_key = keys[node]

    # Wrong/unavailable key is ignored.
    if current_key is None:
        return "ignored"

    if event["key"] != current_key:
        return "ignored"

    state = states.get(node)

    cached = pipeline_cache_entry_reusable(
        cache,
        node,
        current_key,
    )

    # --------------------------------------------------------
    # Already successfully cached
    # --------------------------------------------------------

    if cached:

        if event["status"] == "succeeded":

            if (
                event["artifactDigest"]
                != cache[node]["artifactDigest"]
            ):
                return "evidence_conflict"

            # Exact same successful evidence is dealt with
            # by event-ID replay logic.
            return "status_conflict"

        return "status_conflict"

    # --------------------------------------------------------
    # No current state
    # --------------------------------------------------------

    if state is None:

        if (
            event["status"] == "started"
            and event["attempt"] == 1
        ):

            states[node] = {
                "key": current_key,
                "status": "started",
                "attempt": 1,
                "eventId": event["eventId"],
            }

            return "accepted"

        # Completion or attempt > 1 without start
        # is ignored.
        return "ignored"

    # --------------------------------------------------------
    # State belongs to another key.
    #
    # This is stale state and cannot control the current key.
    # --------------------------------------------------------

    if state.get("key") != current_key:

        if (
            event["status"] == "started"
            and event["attempt"] == 1
        ):

            states[node] = {
                "key": current_key,
                "status": "started",
                "attempt": 1,
                "eventId": event["eventId"],
            }

            return "accepted"

        return "ignored"

    previous_status = state["status"]
    previous_attempt = state["attempt"]

    # --------------------------------------------------------
    # started(n) -> completion(n)
    # --------------------------------------------------------

    if previous_status == "started":

        if event["attempt"] != previous_attempt:
            return "status_conflict"

        if event["status"] in (
            "succeeded",
            "retryable_failed",
            "terminal_failed",
        ):

            if event["status"] == "succeeded":

                states[node] = {
                    "key": current_key,
                    "status": "succeeded",
                    "attempt": event["attempt"],
                    "eventId": event["eventId"],
                }

                # Immutable content-addressed evidence.
                cache[node] = {
                    "key": current_key,
                    "artifactDigest":
                        event["artifactDigest"],
                    "eventId":
                        event["eventId"],
                }

            else:

                states[node] = {
                    "key": current_key,
                    "status": event["status"],
                    "attempt": event["attempt"],
                    "eventId": event["eventId"],
                }

            return "accepted"

        return "status_conflict"

    # --------------------------------------------------------
    # retryable_failed(n) -> started(n+1)
    # --------------------------------------------------------

    if previous_status == "retryable_failed":

        if (
            event["status"] == "started"
            and event["attempt"]
            == previous_attempt + 1
        ):

            states[node] = {
                "key": current_key,
                "status": "started",
                "attempt": event["attempt"],
                "eventId": event["eventId"],
            }

            return "accepted"

        return "status_conflict"

    # --------------------------------------------------------
    # terminal_failed cannot transition
    # --------------------------------------------------------

    if previous_status == "terminal_failed":

        return "status_conflict"

    # --------------------------------------------------------
    # succeeded state
    # --------------------------------------------------------

    if previous_status == "succeeded":

        if event["status"] == "succeeded":

            if (
                event["artifactDigest"]
                != cache.get(node, {}).get(
                    "artifactDigest"
                )
            ):
                return "evidence_conflict"

        return "status_conflict"

    return "status_conflict"


# ------------------------------------------------------------
# Validate request-level inputs
# ------------------------------------------------------------

def pipeline_validate_request(body):

    if not isinstance(body, dict):
        return False

    if "session" not in body:
        return False

    if "revision" not in body:
        return False

    if "inputs" not in body:
        return False

    if "events" not in body:
        return False

    session = body["session"]

    if not pipeline_nonempty_string(session):
        return False

    revision = body["revision"]

    if not pipeline_safe_positive_int(revision):
        return False

    inputs = body["inputs"]

    if not isinstance(inputs, dict):
        return False

    # All 12 required inputs must exist and be non-empty strings.
    for name in PIPELINE_INPUT_NAMES:

        if name not in inputs:
            return False

        if not pipeline_nonempty_string(
            inputs[name]
        ):
            return False

    events = body["events"]

    if not isinstance(events, list):
        return False

    return True


# ------------------------------------------------------------
# Revision fingerprint
# ------------------------------------------------------------

def pipeline_inputs_fingerprint(inputs):
    """
    Includes extra input metadata.

    Thus adding/removing/changing ANY input field changes
    the fingerprint.
    """

    raw = pipeline_canonical_json(
        inputs
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


# ------------------------------------------------------------
# /pipeline
# ------------------------------------------------------------

@app.route("/pipeline", methods=["POST"])
def pipeline():

    body = request.get_json(
        silent=True
    )

    if not pipeline_validate_request(body):
        return (
            json.dumps(
                {"error": "INVALID_REQUEST"},
                separators=(",", ":"),
            ),
            409,
            {"Content-Type": "application/json"},
        )

    session = body["session"]
    revision = body["revision"]
    inputs = body["inputs"]
    events = body["events"]

    input_fingerprint = (
        pipeline_inputs_fingerprint(inputs)
    )

    with PIPELINE_LOCK:

        # ====================================================
        # Create session
        # ====================================================

        stored = PIPELINE_SESSIONS.get(
            session
        )

        if stored is None:

            stored = {
                "revision": revision,
                "inputs": dict(inputs),
                "inputFingerprint":
                    input_fingerprint,

                # Successful content-addressed
                # evidence survives revisions.
                "cache": {},

                # Current revision execution state.
                "states": {},

                # Global event IDs for this session.
                "events": {},
            }

            PIPELINE_SESSIONS[session] = stored

        # ====================================================
        # Existing session
        # ====================================================

        else:

            stored_revision = stored["revision"]

            # ------------------------------------------------
            # Older revision:
            # well-formed events are ignored.
            #
            # But request itself does not replace the current
            # revision.
            # ------------------------------------------------

            if revision < stored_revision:

                keys = pipeline_compute_keys(
                    stored["inputs"],
                    stored["cache"],
                )

                nodes = pipeline_build_nodes(
                    stored["inputs"],
                    stored["cache"],
                    stored["states"],
                    keys,
                )

                return (
                    json.dumps(
                        {
                            "revision":
                                stored_revision,
                            "acceptedEventIds": [],
                            "ignoredEventIds": [],
                            "nodes": nodes,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    200,
                    {
                        "Content-Type":
                            "application/json"
                    },
                )

            # ------------------------------------------------
            # Same revision
            # ------------------------------------------------

            if revision == stored_revision:

                if (
                    stored["inputFingerprint"]
                    != input_fingerprint
                ):

                    return (
                        json.dumps(
                            {
                                "error":
                                    "REVISION_CONFLICT"
                            },
                            separators=(",", ":"),
                        ),
                        409,
                        {
                            "Content-Type":
                                "application/json"
                        },
                    )

            # ------------------------------------------------
            # New revision
            # ------------------------------------------------

            elif revision > stored_revision:

                # Inputs replaced.
                stored["revision"] = revision
                stored["inputs"] = dict(inputs)
                stored["inputFingerprint"] = (
                    input_fingerprint
                )

                # Current attempt/terminal state is cleared.
                #
                # IMPORTANT:
                # successful content-addressed cache remains.
                stored["states"] = {}

        # ====================================================
        # Work on a transaction copy.
        #
        # If ANY event causes a 409, absolutely NOTHING from
        # the batch is committed.
        # ====================================================

        working_cache = {
            node: dict(entry)
            for node, entry
            in stored["cache"].items()
        }

        working_states = {
            node: dict(state)
            for node, state
            in stored["states"].items()
        }

        working_events = dict(
            stored["events"]
        )

        accepted_event_ids = []
        ignored_event_ids = []

        # ====================================================
        # Process events in input order
        # ====================================================

        for event in events:

            # ------------------------------------------------
            # Invalid event
            #
            # Invalid events are ignored, not conflicts.
            # ------------------------------------------------

            if not pipeline_valid_event(event):

                if isinstance(event, dict):
                    eid = event.get("eventId")

                    if pipeline_nonempty_string(eid):
                        ignored_event_ids.append(eid)

                continue

            event_id = event["eventId"]

            canonical = pipeline_event_json(
                event
            )

            # ------------------------------------------------
            # Existing event ID
            # ------------------------------------------------

            if event_id in working_events:

                previous_canonical = (
                    working_events[event_id]
                )

                if previous_canonical == canonical:

                    # Exact replay.
                    ignored_event_ids.append(
                        event_id
                    )

                    continue

                # Same ID, different event.
                #
                # Entire batch rolls back.
                return (
                    json.dumps(
                        {
                            "error":
                                "EVENT_ID_CONFLICT"
                        },
                        separators=(",", ":"),
                    ),
                    409,
                    {
                        "Content-Type":
                            "application/json"
                    },
                )

            # ------------------------------------------------
            # Wrong revision
            # ------------------------------------------------

            if event["revision"] != stored["revision"]:

                ignored_event_ids.append(
                    event_id
                )

                continue

            # ------------------------------------------------
            # Node/key availability and transition
            # ------------------------------------------------

            outcome = pipeline_apply_event(
                event,
                stored["inputs"],
                working_cache,
                working_states,
            )

            if outcome == "evidence_conflict":

                return (
                    json.dumps(
                        {
                            "error":
                                "EVIDENCE_CONFLICT"
                        },
                        separators=(",", ":"),
                    ),
                    409,
                    {
                        "Content-Type":
                            "application/json"
                    },
                )

            if outcome == "status_conflict":

                return (
                    json.dumps(
                        {
                            "error":
                                "STATUS_CONFLICT"
                        },
                        separators=(",", ":"),
                    ),
                    409,
                    {
                        "Content-Type":
                            "application/json"
                    },
                )

            # ------------------------------------------------
            # Ignored event does NOT consume its ID.
            # ------------------------------------------------

            if outcome == "ignored":

                ignored_event_ids.append(
                    event_id
                )

                continue

            # ------------------------------------------------
            # Accepted event consumes its ID.
            # ------------------------------------------------

            if outcome == "accepted":

                working_events[event_id] = (
                    canonical
                )

                accepted_event_ids.append(
                    event_id
                )

        # ====================================================
        # COMMIT TRANSACTION
        # ====================================================

        stored["cache"] = working_cache
        stored["states"] = working_states
        stored["events"] = working_events

        # ====================================================
        # Build deterministic response
        # ====================================================

        keys = pipeline_compute_keys(
            stored["inputs"],
            stored["cache"],
        )

        nodes = pipeline_build_nodes(
            stored["inputs"],
            stored["cache"],
            stored["states"],
            keys,
        )

        response = {
            "revision":
                stored["revision"],
            "acceptedEventIds":
                accepted_event_ids,
            "ignoredEventIds":
                ignored_event_ids,
            "nodes":
                nodes,
        }

        return (
            json.dumps(
                response,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            200,
            {
                "Content-Type":
                    "application/json"
            },
        )

# ============================================================
# Q7 - Publish a Verifiable Model Bundle and Model Card
# Endpoint: POST /verify-bundle
# ============================================================

VERIFY_BUNDLE_REQUIRED_FILES = (
    "README.md",
    "training_manifest.json",
    "evaluation.json",
    "inventory.json",
    "adapter_model.safetensors",
    "adapter_config.json",
)

VERIFY_BUNDLE_UNSAFE_EXTENSIONS = (
    ".bin",
    ".pt",
    ".pth",
    ".pkl",
    ".pickle",
)

VERIFY_BUNDLE_SAFE_MAX = 2**53 - 1


def vb_utf8(value):
    return value.encode("utf-8")


def vb_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def vb_sorted_codes(codes):
    return sorted(
        set(codes),
        key=vb_utf8,
    )


def vb_safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= VERIFY_BUNDLE_SAFE_MAX
    )


def vb_positive_safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= VERIFY_BUNDLE_SAFE_MAX
    )


def vb_finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def vb_unit(value):
    return (
        vb_finite(value)
        and 0 <= float(value) <= 1
    )


def vb_nonempty_string(value):
    return (
        isinstance(value, str)
        and len(value) > 0
    )


def vb_sha256_text(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def vb_valid_sha256(value):
    return (
        isinstance(value, str)
        and re.fullmatch(
            r"[0-9a-f]{64}",
            value,
        ) is not None
    )


def vb_valid_base_revision(value):
    return (
        isinstance(value, str)
        and re.fullmatch(
            r"[0-9a-f]{40}",
            value,
        ) is not None
    )


# ------------------------------------------------------------
# Parse JSON file
# ------------------------------------------------------------

def vb_parse_json(files, filename, violations):
    content = files.get(filename)

    if not isinstance(content, str):
        violations.add(
            "INVALID_FILE:" + filename
        )
        return None

    try:
        return json.loads(content)
    except Exception:
        violations.add(
            "INVALID_JSON:" + filename
        )
        return None


# ------------------------------------------------------------
# Inventory
# ------------------------------------------------------------

def vb_recompute_inventory(files):
    """
    Recompute the exact inventory from all files except
    inventory.json itself.
    """

    inventory = []

    for name, content in files.items():

        if name == "inventory.json":
            continue

        if not isinstance(name, str):
            continue

        if not isinstance(content, str):
            continue

        raw = content.encode("utf-8")

        inventory.append({
            "name": name,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })

    inventory.sort(
        key=lambda item: vb_utf8(item["name"])
    )

    return inventory


def vb_inventory_digest(inventory):
    return hashlib.sha256(
        vb_json(inventory).encode("utf-8")
    ).hexdigest()


def vb_validate_inventory(
    files,
    inventory_value,
    violations,
):
    """
    Validate inventory.json against the actual files.
    """

    actual = vb_recompute_inventory(files)

    if not isinstance(inventory_value, list):

        violations.add(
            "INVALID_JSON:inventory.json"
        )

        return actual, None

    normalized = []

    seen = set()

    valid_structure = True

    for entry in inventory_value:

        if not isinstance(entry, dict):

            valid_structure = False
            break

        if set(entry.keys()) != {
            "name",
            "bytes",
            "sha256",
        }:

            valid_structure = False
            break

        name = entry["name"]
        byte_count = entry["bytes"]
        digest = entry["sha256"]

        if (
            not isinstance(name, str)
            or name == ""
            or name in seen
        ):

            valid_structure = False
            break

        if not vb_safe_integer(byte_count):

            valid_structure = False
            break

        if not vb_valid_sha256(digest):

            valid_structure = False
            break

        seen.add(name)

        normalized.append({
            "name": name,
            "bytes": byte_count,
            "sha256": digest,
        })

    if not valid_structure:

        violations.add(
            "INVENTORY_MISMATCH"
        )

        return actual, vb_inventory_digest(actual)

    # Inventory must already be sorted by UTF-8 filename.
    expected_sorted = sorted(
        normalized,
        key=lambda item: vb_utf8(item["name"])
    )

    if normalized != expected_sorted:

        violations.add(
            "INVENTORY_MISMATCH"
        )

    # Compare exact entries.
    if normalized != actual:

        violations.add(
            "INVENTORY_MISMATCH"
        )

    actual_names = {
        item["name"]
        for item in actual
    }

    recorded_names = {
        item["name"]
        for item in normalized
    }

    # Files which exist but aren't tracked.
    if actual_names - recorded_names:

        violations.add(
            "UNTRACKED_FILE"
        )

    digest = vb_inventory_digest(actual)

    return actual, digest


# ------------------------------------------------------------
# Adapter config
# ------------------------------------------------------------

def vb_validate_adapter_config(
    config,
    violations,
):

    if not isinstance(config, dict):

        violations.add(
            "INVALID_ADAPTER_CONFIG"
        )

        return

    r = config.get("r")

    targets = config.get(
        "target_modules"
    )

    if not vb_positive_safe_integer(r):

        violations.add(
            "INVALID_ADAPTER_CONFIG"
        )

    if not isinstance(targets, list):

        violations.add(
            "INVALID_ADAPTER_CONFIG"
        )

        return

    if len(targets) == 0:

        violations.add(
            "INVALID_ADAPTER_CONFIG"
        )

        return

    if any(
        not isinstance(x, str)
        or x == ""
        for x in targets
    ):

        violations.add(
            "INVALID_ADAPTER_CONFIG"
        )

    if len(set(targets)) != len(targets):

        violations.add(
            "INVALID_ADAPTER_CONFIG"
        )


# ------------------------------------------------------------
# Training manifest
# ------------------------------------------------------------

def vb_validate_training_manifest(
    manifest,
    model_digest,
    evaluation_digest,
    violations,
):

    if not isinstance(manifest, dict):

        violations.add(
            "INVALID_TRAINING_MANIFEST"
        )

        return None

    required = (
        "baseRevision",
        "task",
        "datasetDigest",
        "codeDigest",
        "trainingConfigDigest",
        "modelArtifactDigest",
        "evaluationArtifactDigest",
    )

    for field in required:

        if (
            field not in manifest
            or not vb_nonempty_string(
                manifest[field]
            )
        ):

            violations.add(
                "MISSING_MANIFEST_FIELD:" + field
            )

    base_revision = manifest.get(
        "baseRevision"
    )

    if (
        base_revision is not None
        and (
            not isinstance(base_revision, str)
            or re.fullmatch(
                r"[0-9a-f]{40}",
                base_revision,
            ) is None
        )
    ):

        violations.add(
            "MUTABLE_BASE_REVISION"
        )

    # Verify model artifact digest.
    if (
        isinstance(
            manifest.get(
                "modelArtifactDigest"
            ),
            str,
        )
        and manifest.get(
            "modelArtifactDigest"
        ) != model_digest
    ):

        violations.add(
            "MODEL_ARTIFACT_MISMATCH"
        )

    # Verify evaluation artifact digest.
    if (
        isinstance(
            manifest.get(
                "evaluationArtifactDigest"
            ),
            str,
        )
        and manifest.get(
            "evaluationArtifactDigest"
        ) != evaluation_digest
    ):

        violations.add(
            "EVALUATION_ARTIFACT_MISMATCH"
        )

    return manifest


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def vb_validate_evaluation(
    evaluation,
    model_digest,
    policy,
    violations,
):

    if not isinstance(evaluation, dict):

        violations.add(
            "INVALID_EVALUATION"
        )

        return

    # --------------------------------------------------------
    # Model binding
    # --------------------------------------------------------

    evaluation_model_digest = evaluation.get(
        "modelArtifactDigest"
    )

    if (
        not isinstance(
            evaluation_model_digest,
            str,
        )
        or evaluation_model_digest
        != model_digest
    ):

        violations.add(
            "MODEL_ARTIFACT_MISMATCH"
        )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    aggregate = evaluation.get(
        "accuracy"
    )

    if not vb_finite(aggregate):

        violations.add(
            "INVALID_AGGREGATE"
        )

    elif not vb_unit(aggregate):

        violations.add(
            "INVALID_AGGREGATE"
        )

    # --------------------------------------------------------
    # Required slices
    # --------------------------------------------------------

    slices = evaluation.get(
        "slices"
    )

    if not isinstance(slices, dict):

        for name in policy["requiredSlices"]:

            violations.add(
                "MISSING_SLICE:" + name
            )

        return

    for name in policy["requiredSlices"]:

        if name not in slices:

            violations.add(
                "MISSING_SLICE:" + name
            )

            continue

        value = slices[name]

        if not vb_finite(value):

            violations.add(
                "SLICE_RANGE:" + name
            )

        elif not vb_unit(value):

            violations.add(
                "SLICE_RANGE:" + name
            )


# ------------------------------------------------------------
# Model-card marker
# ------------------------------------------------------------

def vb_find_model_card(readme):
    """
    Find the exact marker:

    <!-- tds-model-card {...} -->

    JSON braces inside strings are handled naturally by
    locating the marker prefix and the closing -->.
    """

    prefix = "<!-- tds-model-card "

    count = readme.count(prefix)

    if count == 0:
        return 0, None

    if count > 1:
        return count, None

    start = readme.find(prefix)

    payload_start = (
        start + len(prefix)
    )

    end = readme.find(
        "-->",
        payload_start,
    )

    if end == -1:
        return 1, None

    payload = readme[
        payload_start:end
    ]

    try:
        parsed = json.loads(payload)
    except Exception:
        return 1, "__INVALID__"

    if not isinstance(parsed, dict):
        return 1, "__INVALID__"

    return 1, parsed


# ------------------------------------------------------------
# Main verifier
# ------------------------------------------------------------

def verify_bundle_compute(body):

    violations = set()

    policy = body.get(
        "policy"
    )

    files = body.get(
        "files"
    )

    # --------------------------------------------------------
    # Policy
    # --------------------------------------------------------

    policy_valid = True

    if not isinstance(policy, dict):

        return {
            "decision": "reject",
            "violations": [
                "INVALID_POLICY"
            ],
            "inventoryDigest": None,
        }

    required_slices = policy.get(
        "requiredSlices"
    )

    if (
        not isinstance(
            required_slices,
            list,
        )
        or len(required_slices) == 0
        or any(
            not isinstance(x, str)
            or x == ""
            for x in required_slices
        )
        or len(set(required_slices))
        != len(required_slices)
    ):

        violations.add(
            "INVALID_POLICY"
        )

    for field in (
        "license",
        "intendedUse",
        "limitations",
    ):

        if not vb_nonempty_string(
            policy.get(field)
        ):

            violations.add(
                "INVALID_POLICY"
            )

    if not isinstance(files, dict):

        return {
            "decision": "reject",
            "violations": [
                "INVALID_POLICY"
            ],
            "inventoryDigest": None,
        }

    # --------------------------------------------------------
    # File names and UTF-8 content
    # --------------------------------------------------------

    for filename, content in files.items():

        if (
            not isinstance(filename, str)
            or filename == ""
            or not isinstance(content, str)
        ):

            violations.add(
                "INVALID_FILE:" + str(filename)
            )

            continue

        lower_name = filename.lower()

        for extension in (
            VERIFY_BUNDLE_UNSAFE_EXTENSIONS
        ):

            if lower_name.endswith(extension):

                violations.add(
                    "UNSAFE_WEIGHTS"
                )

                break

    # --------------------------------------------------------
    # Required files
    # --------------------------------------------------------

    for filename in (
        VERIFY_BUNDLE_REQUIRED_FILES
    ):

        if filename not in files:

            violations.add(
                "MISSING_FILE:" + filename
            )

    # --------------------------------------------------------
    # Inventory
    # --------------------------------------------------------

    inventory_digest = None

    inventory_value = None

    if "inventory.json" in files:

        inventory_value = vb_parse_json(
            files,
            "inventory.json",
            violations,
        )

        actual_inventory, inventory_digest = (
            vb_validate_inventory(
                files,
                inventory_value,
                violations,
            )
        )

    # --------------------------------------------------------
    # Adapter config
    # --------------------------------------------------------

    adapter_config = None

    if "adapter_config.json" in files:

        adapter_config = vb_parse_json(
            files,
            "adapter_config.json",
            violations,
        )

        if adapter_config is not None:

            vb_validate_adapter_config(
                adapter_config,
                violations,
            )

    # --------------------------------------------------------
    # Artifact digests
    # --------------------------------------------------------

    model_digest = None

    evaluation_digest = None

    if isinstance(
        files.get(
            "adapter_model.safetensors"
        ),
        str,
    ):

        model_digest = vb_sha256_text(
            files[
                "adapter_model.safetensors"
            ]
        )

    if isinstance(
        files.get(
            "evaluation.json"
        ),
        str,
    ):

        evaluation_digest = vb_sha256_text(
            files["evaluation.json"]
        )

    # --------------------------------------------------------
    # Training manifest
    # --------------------------------------------------------

    manifest = None

    if "training_manifest.json" in files:

        manifest = vb_parse_json(
            files,
            "training_manifest.json",
            violations,
        )

        if manifest is not None:

            vb_validate_training_manifest(
                manifest,
                model_digest,
                evaluation_digest,
                violations,
            )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    evaluation = None

    if "evaluation.json" in files:

        evaluation = vb_parse_json(
            files,
            "evaluation.json",
            violations,
        )

        if evaluation is not None:

            vb_validate_evaluation(
                evaluation,
                model_digest,
                {
                    "requiredSlices":
                        required_slices
                    if isinstance(
                        required_slices,
                        list,
                    )
                    else [],
                },
                violations,
            )

    # --------------------------------------------------------
    # Model card
    # --------------------------------------------------------

    if "README.md" in files:

        readme = files["README.md"]

        if isinstance(readme, str):

            marker_count, card = (
                vb_find_model_card(
                    readme
                )
            )

            if marker_count == 0:

                violations.add(
                    "MODEL_CARD_COUNT"
                )

                violations.add(
                    "MISSING_MODEL_CARD"
                )

            elif marker_count > 1:

                # Multiple markers emit ONLY
                # MODEL_CARD_COUNT.
                violations.add(
                    "MODEL_CARD_COUNT"
                )

            elif card == "__INVALID__":

                violations.add(
                    "INVALID_MODEL_CARD"
                )

            elif isinstance(card, dict):

                # Compare only when machine evidence
                # exists.
                expected = {
                    "task":
                        manifest.get("task")
                        if isinstance(
                            manifest,
                            dict,
                        )
                        else None,

                    "baseRevision":
                        manifest.get(
                            "baseRevision"
                        )
                        if isinstance(
                            manifest,
                            dict,
                        )
                        else None,

                    "datasetDigest":
                        manifest.get(
                            "datasetDigest"
                        )
                        if isinstance(
                            manifest,
                            dict,
                        )
                        else None,

                    "modelArtifactDigest":
                        manifest.get(
                            "modelArtifactDigest"
                        )
                        if isinstance(
                            manifest,
                            dict,
                        )
                        else None,

                    "license":
                        policy.get(
                            "license"
                        ),

                    "intendedUse":
                        policy.get(
                            "intendedUse"
                        ),

                    "limitations":
                        policy.get(
                            "limitations"
                        ),
                }

                for field, expected_value in (
                    expected.items()
                ):

                    if (
                        card.get(field)
                        != expected_value
                    ):

                        violations.add(
                            "MODEL_CARD_MISMATCH"
                        )

                        break

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    final_codes = vb_sorted_codes(
        violations
    )

    return {
        "decision":
            "admit"
            if not final_codes
            else "reject",

        "violations":
            final_codes,

        "inventoryDigest":
            inventory_digest,
    }


# ------------------------------------------------------------
# POST /verify-bundle
# ------------------------------------------------------------

@app.route(
    "/verify-bundle",
    methods=["POST"],
)
def verify_bundle():

    body = request.get_json(
        silent=True
    )

    # --------------------------------------------------------
    # Only these malformed top-level requests require
    # HTTP 400 with exactly INVALID_INPUT.
    # --------------------------------------------------------

    if not isinstance(body, dict):

        return (
            json.dumps(
                {"error": "INVALID_INPUT"},
                separators=(",", ":"),
            ),
            400,
            {
                "Content-Type":
                    "application/json"
            },
        )

    if (
        "policy" not in body
        or "files" not in body
        or not isinstance(
            body.get("policy"),
            dict,
        )
        or not isinstance(
            body.get("files"),
            dict,
        )
    ):

        return (
            json.dumps(
                {"error": "INVALID_INPUT"},
                separators=(",", ":"),
            ),
            400,
            {
                "Content-Type":
                    "application/json"
            },
        )

    result = verify_bundle_compute(
        body
    )

    return app.response_class(
        response=vb_json(result),
        status=200,
        mimetype="application/json",
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