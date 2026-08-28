from flask import Flask, request, jsonify
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import unicodedata
from copy import deepcopy

app = Flask(__name__)

SAFE_MAX = 9007199254740991

# Q2 persistent in-memory run storage
RUNS = {}

TS_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})$"
)

HEX8_RE = re.compile(r"^[0-9a-f]{8}$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
GEN_RE = re.compile(r"^[0-9]+$")
URI_RE = re.compile(r"^gs://[^/\s]+/.+$")


# ----------------------------
# Shared helper functions
# ----------------------------

def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def utf8_key(value):
    return str(value).encode("utf-8")


def utf8_sort(values):
    return sorted(values, key=utf8_key)


def sorted_codes(codes):
    return sorted(set(codes), key=utf8_key)


def safe_int(value, nonnegative=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    if abs(value) > SAFE_MAX:
        return False
    if positive:
        return value > 0
    if nonnegative:
        return value >= 0
    return True


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def parse_timestamp(value):
    """Strict timestamp validation and UTC conversion."""
    if not isinstance(value, str):
        return None

    match = TS_RE.fullmatch(value)
    if not match:
        return None

    year, month, day, hour, minute, second, fraction, offset = match.groups()

    if offset != "Z":
        offset_hour = int(offset[1:3])
        offset_minute = int(offset[4:6])

        if offset_hour > 14 or offset_minute > 59:
            return None
        if offset_hour == 14 and offset_minute != 0:
            return None

    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        return datetime.fromisoformat(normalized).astimezone(timezone.utc)
    except ValueError:
        return None


def utc_milliseconds(dt):
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ----------------------------
# CRC32C implementation (Q1)
# ----------------------------

CRC32C_TABLE = []


def build_crc32c_table():
    poly = 0x82F63B78

    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ poly if (crc & 1) else crc >> 1
        CRC32C_TABLE.append(crc & 0xFFFFFFFF)


build_crc32c_table()


def crc32c(data):
    crc = 0xFFFFFFFF

    for byte in data:
        crc = CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)

    return f"{(crc ^ 0xFFFFFFFF) & 0xFFFFFFFF:08x}"


# ============================================================
# Q1: Build Immutable, Leakage-Safe Training Corpus
# ============================================================

def canonical_text(value):
    value = unicodedata.normalize("NFKC", value)
    value = value.lower().strip()
    return " ".join(value.split())


def word_set(text):
    """
    Unicode letter/number word set.
    Characters that are neither letters nor numbers delimit words.
    """
    words = []
    current = []

    for ch in text.lower():
        category = unicodedata.category(ch)
        if category[0] in ("L", "N"):
            current.append(ch)
        else:
            if current:
                words.append("".join(current))
                current = []

    if current:
        words.append("".join(current))

    return set(words)


def jaccard_similarity(a, b):
    if not a and not b:
        return 1.0

    union = a | b
    return len(a & b) / len(union)


def valid_jsonl_row(row):
    expected = {"id", "entity", "eventTime", "revision", "text"}

    if not isinstance(row, dict) or set(row.keys()) != expected:
        return False

    if not all(isinstance(row[field], str) for field in ("id", "entity", "eventTime", "text")):
        return False

    if not safe_int(row["revision"], nonnegative=True):
        return False

    return parse_timestamp(row["eventTime"]) is not None


def parse_jsonl(content):
    """
    Returns (rows, schema_invalid, jsonl_invalid).
    Blank lines are ignored.
    """
    rows = []
    schema_invalid = False
    jsonl_invalid = False

    for line in content.splitlines():
        if line.strip() == "":
            continue

        try:
            item = json.loads(line)
        except Exception:
            jsonl_invalid = True
            continue

        if not valid_jsonl_row(item):
            schema_invalid = True
        else:
            rows.append(item)

    if not rows and not jsonl_invalid:
        schema_invalid = True

    return rows, schema_invalid, jsonl_invalid


def empty_corpus_response():
    empty_hash = hashlib.sha256(b"").hexdigest()

    return {
        "splits": {
            "train": [],
            "validation": [],
            "test": []
        },
        "rejectedObjects": [],
        "rejectedRows": [],
        "digests": {
            "train": empty_hash,
            "validation": empty_hash,
            "test": empty_hash
        },
        "lineage": []
    }


def valid_policy(policy):
    if not isinstance(policy, dict):
        return None

    if set(policy.keys()) != {
        "minTime",
        "maxTime",
        "contaminationThreshold"
    }:
        return None

    min_time = parse_timestamp(policy["minTime"])
    max_time = parse_timestamp(policy["maxTime"])
    threshold = policy["contaminationThreshold"]

    if min_time is None or max_time is None:
        return None

    if min_time > max_time:
        return None

    if not finite_number(threshold) or not 0 <= threshold <= 1:
        return None

    return min_time, max_time, threshold


def canonicalize_row(row):
    dt = parse_timestamp(row["eventTime"])

    return {
        "id": row["id"],
        "entity": canonical_text(row["entity"]),
        "eventTime": utc_milliseconds(dt),
        "revision": row["revision"],
        "text": canonical_text(row["text"])
    }


def row_json_for_sort(row):
    return compact_json({
        "id": row["id"],
        "entity": row["entity"],
        "eventTime": row["eventTime"],
        "revision": row["revision"],
        "text": row["text"]
    })


def split_digest(rows):
    blob = "".join(row_json_for_sort(row) + "\n" for row in rows)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@app.post("/build-corpus")
def build_corpus():
    data = request.get_json(silent=True)

    # Required HTTP 400 condition
    if (
        not isinstance(data, dict)
        or "policy" not in data
        or not isinstance(data.get("objects"), list)
    ):
        return jsonify({"error": "INVALID_INPUT"}), 400

    response = empty_corpus_response()
    policy_info = valid_policy(data.get("policy"))

    all_valid_rows = []

    # Validate each supplied object independently
    for obj in data["objects"]:
        codes = []
        uri = obj.get("uri") if isinstance(obj, dict) and isinstance(obj.get("uri"), str) else None

        if not isinstance(obj, dict):
            response["rejectedObjects"].append({
                "uri": None,
                "reasonCodes": ["URI_INVALID", "GENERATION_INVALID", "SCHEMA_INVALID"]
            })
            continue

        object_uri = obj.get("uri")

        if not isinstance(object_uri, str) or not URI_RE.fullmatch(object_uri):
            codes.append("URI_INVALID")

        generation = obj.get("generation")
        fetched_generation = obj.get("fetchedGeneration")

        generation_valid = isinstance(generation, str) and GEN_RE.fullmatch(generation)
        fetched_valid = (
            isinstance(fetched_generation, str)
            and GEN_RE.fullmatch(fetched_generation)
        )

        if not generation_valid or not fetched_valid:
            codes.append("GENERATION_INVALID")

        if generation != fetched_generation:
            codes.append("GENERATION_MISMATCH")

        provided_crc = obj.get("crc32c")
        content = obj.get("content")

        crc_valid = isinstance(provided_crc, str) and HEX8_RE.fullmatch(provided_crc)

        if not crc_valid:
            codes.append("CRC32C_INVALID")

        if isinstance(content, str) and crc_valid:
            actual_crc = crc32c(content.encode("utf-8"))
            if actual_crc != provided_crc:
                codes.append("CRC32C_MISMATCH")

        schema_id = obj.get("schemaId")
        if schema_id != "training-v1":
            codes.append("SCHEMA_INVALID")

        parsed_rows = []
        if not isinstance(content, str):
            codes.append("SCHEMA_INVALID")
        else:
            parsed_rows, schema_invalid, jsonl_invalid = parse_jsonl(content)

            if schema_invalid:
                codes.append("SCHEMA_INVALID")
            if jsonl_invalid:
                codes.append("JSONL_INVALID")

        if codes:
            response["rejectedObjects"].append({
                "uri": uri,
                "reasonCodes": sorted_codes(codes)
            })
            continue

        lineage_entry = {
            "uri": object_uri,
            "generation": generation,
            "crc32c": provided_crc,
            "schemaId": schema_id
        }
        response["lineage"].append(lineage_entry)

        for row in parsed_rows:
            all_valid_rows.append(canonicalize_row(row))

    # Deduplicate canonical rows
    grouped = {}

    for row in all_valid_rows:
        key = compact_json([row["entity"], row["eventTime"], row["text"]])
        grouped.setdefault(key, []).append(row)

    retained_rows = []

    for group_rows in grouped.values():
        winner = sorted(
            group_rows,
            key=lambda r: (-r["revision"], utf8_key(r["id"]))
        )[0]

        retained_rows.append(winner)

        for loser in group_rows:
            if loser is not winner:
                response["rejectedRows"].append({
                    "id": loser["id"],
                    "reasonCodes": ["DUPLICATE"]
                })

    # Invalid policy rejects every retained row
    if policy_info is None:
        for row in retained_rows:
            response["rejectedRows"].append({
                "id": row["id"],
                "reasonCodes": ["POLICY_INVALID"]
            })

        retained_rows = []

    else:
        min_time, max_time, threshold = policy_info
        window_rows = []

        for row in retained_rows:
            event_dt = parse_timestamp(row["eventTime"])

            if event_dt < min_time or event_dt > max_time:
                response["rejectedRows"].append({
                    "id": row["id"],
                    "reasonCodes": ["OUT_OF_WINDOW"]
                })
            else:
                window_rows.append(row)

        retained_rows = window_rows

        # Deterministic split based on SHA256 first byte
        provisional_splits = {
            "train": [],
            "validation": [],
            "test": []
        }

        for row in retained_rows:
            bucket = hashlib.sha256(
                row["entity"].encode("utf-8")
            ).digest()[0] % 10

            if bucket <= 5:
                provisional_splits["train"].append(row)
            elif bucket <= 7:
                provisional_splits["validation"].append(row)
            else:
                provisional_splits["test"].append(row)

        train_word_sets = [
            word_set(row["text"])
            for row in provisional_splits["train"]
        ]

        # Keep all train rows. Remove contaminated validation/test rows.
        response["splits"]["train"] = provisional_splits["train"]

        for split_name in ("validation", "test"):
            for row in provisional_splits[split_name]:
                candidate_words = word_set(row["text"])

                contaminated = any(
                    jaccard_similarity(candidate_words, train_words) >= threshold
                    for train_words in train_word_sets
                )

                if contaminated:
                    response["rejectedRows"].append({
                        "id": row["id"],
                        "reasonCodes": ["TRAIN_CONTAMINATION"]
                    })
                else:
                    response["splits"][split_name].append(row)

    # Required deterministic split ordering
    for split_name in ("train", "validation", "test"):
        response["splits"][split_name] = sorted(
            response["splits"][split_name],
            key=lambda r: (utf8_key(r["id"]), row_json_for_sort(r).encode("utf-8"))
        )

    # Deterministic rejection/lineage ordering
    response["rejectedObjects"] = sorted(
        response["rejectedObjects"],
        key=lambda x: (
            utf8_key(x["uri"]) if isinstance(x["uri"], str) else b"",
            compact_json(x).encode("utf-8")
        )
    )

    response["rejectedRows"] = sorted(
        response["rejectedRows"],
        key=lambda x: (
            utf8_key(x["id"]),
            compact_json(x).encode("utf-8")
        )
    )

    response["lineage"] = sorted(
        response["lineage"],
        key=lambda x: (
            utf8_key(x["uri"]),
            compact_json(x).encode("utf-8")
        )
    )

    for item in response["rejectedObjects"]:
        item["reasonCodes"] = sorted_codes(item["reasonCodes"])

    for item in response["rejectedRows"]:
        item["reasonCodes"] = sorted_codes(item["reasonCodes"])

    response["digests"] = {
        "train": split_digest(response["splits"]["train"]),
        "validation": split_digest(response["splits"]["validation"]),
        "test": split_digest(response["splits"]["test"])
    }

    return jsonify(response), 200


# ============================================================
# Q2: Leakage-Safe BigQuery ML Experiment
# ============================================================

def valid_run_id(run_id):
    return isinstance(run_id, str) and 1 <= len(run_id) <= 128


def invalid_select_response(run_id):
    return {
        "runId": run_id if isinstance(run_id, str) else None,
        "selectedTrialId": None,
        "trainRowIds": [],
        "evalRowIds": [],
        "featureNames": [],
        "datasetDigest": None,
        "reasonCodes": ["INVALID_INPUT"]
    }


def valid_selection(data):
    expected = {
        "phase",
        "runId",
        "forbiddenFeatures",
        "numTrialsLimit",
        "rows",
        "trials"
    }

    if not isinstance(data, dict) or set(data.keys()) != expected:
        return False

    if data["phase"] != "select" or not valid_run_id(data["runId"]):
        return False

    if not isinstance(data["forbiddenFeatures"], list):
        return False

    if any(not isinstance(name, str) for name in data["forbiddenFeatures"]):
        return False

    if not safe_int(data["numTrialsLimit"], positive=True):
        return False

    if not isinstance(data["rows"], list) or len(data["rows"]) == 0:
        return False

    if not isinstance(data["trials"], list):
        return False

    row_ids = set()

    for row in data["rows"]:
        required_row = {
            "id",
            "entity",
            "eventTime",
            "predictionTime",
            "version",
            "split",
            "features"
        }

        if not isinstance(row, dict) or set(row.keys()) != required_row:
            return False

        if not isinstance(row["id"], str) or row["id"] in row_ids:
            return False
        row_ids.add(row["id"])

        if not isinstance(row["entity"], str):
            return False

        if parse_timestamp(row["eventTime"]) is None:
            return False

        if parse_timestamp(row["predictionTime"]) is None:
            return False

        if not safe_int(row["version"], nonnegative=True):
            return False

        if row["split"] not in ("TRAIN", "EVAL"):
            return False

        if not isinstance(row["features"], dict):
            return False

        for feature_name, feature_data in row["features"].items():
            if not isinstance(feature_name, str):
                return False

            if not isinstance(feature_data, dict):
                return False

            if set(feature_data.keys()) != {"value", "availableAt"}:
                return False

            if parse_timestamp(feature_data["availableAt"]) is None:
                return False

    trial_ids = set()

    for trial in data["trials"]:
        if not isinstance(trial, dict):
            return False

        if set(trial.keys()) != {"trialId", "status", "evalMetric"}:
            return False

        if not safe_int(trial["trialId"], nonnegative=True):
            return False

        if trial["trialId"] in trial_ids:
            return False
        trial_ids.add(trial["trialId"])

        if trial["status"] not in ("SUCCEEDED", "FAILED"):
            return False

    return True


def retained_selection_rows(rows):
    """Deduplicate by [entity, UTC(eventTime)]."""
    grouped = {}

    for row in rows:
        event_time = utc_milliseconds(parse_timestamp(row["eventTime"]))
        key = compact_json([row["entity"], event_time])
        grouped.setdefault(key, []).append(row)

    retained = []

    for candidates in grouped.values():
        retained.append(
            sorted(
                candidates,
                key=lambda row: (-row["version"], utf8_key(row["id"]))
            )[0]
        )

    return retained


def eligible_features(rows, forbidden_features):
    if not rows:
        return []

    names = set(rows[0]["features"].keys())
    forbidden = set(forbidden_features)

    for row in rows[1:]:
        names &= set(row["features"].keys())

    accepted = []

    for name in names:
        if name in forbidden:
            continue

        point_in_time_valid = True

        for row in rows:
            available_at = parse_timestamp(row["features"][name]["availableAt"])
            prediction_time = parse_timestamp(row["predictionTime"])

            if available_at > prediction_time:
                point_in_time_valid = False
                break

        if point_in_time_valid:
            accepted.append(name)

    return utf8_sort(accepted)


def select_trial(data):
    run_id = data.get("runId") if isinstance(data, dict) else None
    canonical_request = compact_json(data)

    # Exact replay / conflict handling
    if valid_run_id(run_id) and run_id in RUNS:
        if RUNS[run_id]["request"] == canonical_request:
            return RUNS[run_id]["response"], 200
        return {"error": "RUN_ID_CONFLICT"}, 409

    if not valid_selection(data):
        response = invalid_select_response(run_id)

        if valid_run_id(run_id):
            RUNS[run_id] = {
                "request": canonical_request,
                "response": deepcopy(response)
            }

        return response, 200

    reason_codes = []

    if len(data["trials"]) > data["numTrialsLimit"]:
        reason_codes.append("TRIAL_LIMIT_EXCEEDED")

    retained_rows = retained_selection_rows(data["rows"])

    train_ids = utf8_sort([
        row["id"]
        for row in retained_rows
        if row["split"] == "TRAIN"
    ])

    eval_ids = utf8_sort([
        row["id"]
        for row in retained_rows
        if row["split"] == "EVAL"
    ])

    features = eligible_features(
        retained_rows,
        data["forbiddenFeatures"]
    )

    successful_trials = [
        trial
        for trial in data["trials"]
        if trial["status"] == "SUCCEEDED"
        and finite_number(trial["evalMetric"])
    ]

    if not successful_trials:
        reason_codes.append("NO_SUCCESSFUL_TRIAL")

    selected_trial = None

    if not reason_codes:
        selected_trial = sorted(
            successful_trials,
            key=lambda trial: (
                -trial["evalMetric"],
                trial["trialId"]
            )
        )[0]["trialId"]

    digest_payload = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": features
    }

    dataset_digest = hashlib.sha256(
        compact_json(digest_payload).encode("utf-8")
    ).hexdigest()

    response = {
        "runId": data["runId"],
        "selectedTrialId": selected_trial,
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": features,
        "datasetDigest": dataset_digest,
        "reasonCodes": sorted_codes(reason_codes)
    }

    if response["reasonCodes"]:
        response["selectedTrialId"] = None

    RUNS[data["runId"]] = {
        "request": canonical_request,
        "response": deepcopy(response)
    }

    return response, 200


def valid_evaluation_shape(data):
    expected = {
        "phase",
        "runId",
        "selectedTrialId",
        "datasetDigest",
        "metricFloor",
        "requiredSlices",
        "rows",
        "bytesProcessed",
        "maxBytes"
    }

    if not isinstance(data, dict) or set(data.keys()) != expected:
        return False

    if data["phase"] != "evaluate":
        return False

    if not valid_run_id(data["runId"]):
        return False

    if not safe_int(data["selectedTrialId"], nonnegative=True):
        return False

    if not isinstance(data["datasetDigest"], str):
        return False

    if not HEX64_RE.fullmatch(data["datasetDigest"]):
        return False

    if not finite_number(data["metricFloor"]):
        return False

    if not 0 <= data["metricFloor"] <= 1:
        return False

    if not isinstance(data["requiredSlices"], dict):
        return False

    for name, floor in data["requiredSlices"].items():
        if not isinstance(name, str) or name == "":
            return False

        if not finite_number(floor) or not 0 <= floor <= 1:
            return False

    if not isinstance(data["rows"], list):
        return False

    if not safe_int(data["bytesProcessed"], nonnegative=True):
        return False

    if not safe_int(data["maxBytes"], nonnegative=True):
        return False

    return True


def valid_test_row(row):
    return (
        isinstance(row, dict)
        and set(row.keys()) == {"label", "prediction", "slice"}
        and isinstance(row["label"], int)
        and not isinstance(row["label"], bool)
        and row["label"] in (0, 1)
        and isinstance(row["prediction"], int)
        and not isinstance(row["prediction"], bool)
        and row["prediction"] in (0, 1)
        and isinstance(row["slice"], str)
        and row["slice"] != ""
    )


def evaluate_trial(data):
    supplied_run_id = data.get("runId") if isinstance(data, dict) else None
    supplied_trial_id = data.get("selectedTrialId") if isinstance(data, dict) else None
    supplied_digest = data.get("datasetDigest") if isinstance(data, dict) else None
    supplied_bytes = data.get("bytesProcessed") if isinstance(data, dict) else None

    response = {
        "runId": supplied_run_id if isinstance(supplied_run_id, str) else None,
        "selectedTrialId": (
            supplied_trial_id
            if safe_int(supplied_trial_id, nonnegative=True)
            else None
        ),
        "datasetDigest": (
            supplied_digest
            if isinstance(supplied_digest, str)
            else None
        ),
        "testMetric": None,
        "criticalSlicePass": False,
        "decision": "reject",
        "bytesProcessed": (
            supplied_bytes
            if safe_int(supplied_bytes, nonnegative=True)
            else None
        ),
        "reasonCodes": []
    }

    if not valid_evaluation_shape(data):
        response["reasonCodes"] = ["INVALID_INPUT"]
        return response, 200

    reason_codes = []

    stored = RUNS.get(data["runId"])

    lineage_valid = (
        stored is not None
        and stored["response"]["selectedTrialId"] is not None
        and stored["response"]["selectedTrialId"] == data["selectedTrialId"]
        and stored["response"]["datasetDigest"] == data["datasetDigest"]
    )

    if not lineage_valid:
        reason_codes.append("INVALID_LINEAGE")

    all_rows_valid = (
        len(data["rows"]) > 0
        and all(valid_test_row(row) for row in data["rows"])
    )

    if not all_rows_valid:
        reason_codes.append("INVALID_TEST_ROW")

    if data["bytesProcessed"] > data["maxBytes"]:
        reason_codes.append("BYTE_LIMIT")

    if all_rows_valid:
        correct = sum(
            row["label"] == row["prediction"]
            for row in data["rows"]
        )

        test_metric = round(correct / len(data["rows"]), 12)
        response["testMetric"] = test_metric

        if test_metric < data["metricFloor"]:
            reason_codes.append("AGGREGATE_FLOOR")

        slice_rows = {}

        for row in data["rows"]:
            slice_rows.setdefault(row["slice"], []).append(row)

        all_slice_gates_pass = True

        for required_slice, floor in data["requiredSlices"].items():
            if required_slice not in slice_rows:
                reason_codes.append(f"MISSING_SLICE:{required_slice}")
                all_slice_gates_pass = False
                continue

            rows_for_slice = slice_rows[required_slice]
            slice_accuracy = round(
                sum(
                    row["label"] == row["prediction"]
                    for row in rows_for_slice
                ) / len(rows_for_slice),
                12
            )

            if slice_accuracy < floor:
                reason_codes.append(f"SLICE_FLOOR:{required_slice}")
                all_slice_gates_pass = False

        # Defined specifically for critical required slice
        if "critical" in data["requiredSlices"]:
            if "critical" in slice_rows:
                critical_rows = slice_rows["critical"]
                critical_accuracy = round(
                    sum(
                        row["label"] == row["prediction"]
                        for row in critical_rows
                    ) / len(critical_rows),
                    12
                )

                response["criticalSlicePass"] = (
                    critical_accuracy >= data["requiredSlices"]["critical"]
                )
            else:
                response["criticalSlicePass"] = False
        else:
            response["criticalSlicePass"] = True

    response["reasonCodes"] = sorted_codes(reason_codes)
    response["decision"] = (
        "admit"
        if len(response["reasonCodes"]) == 0
        else "reject"
    )

    # Requirement: false with invalid lineage/input/test rows/missing or failed slice.
    if (
        not lineage_valid
        or not all_rows_valid
        or any(
            code.startswith("MISSING_SLICE:")
            or code.startswith("SLICE_FLOOR:")
            for code in response["reasonCodes"]
        )
    ):
        response["criticalSlicePass"] = False

    return response, 200


@app.post("/bqml")
def bqml():
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or data.get("phase") not in {"select", "evaluate"}:
        return jsonify({"error": "INVALID_INPUT"}), 400

    if data["phase"] == "select":
        response, status = select_trial(data)
        return jsonify(response), status

    response, status = evaluate_trial(data)
    return jsonify(response), status


@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "endpoints": [
            "POST /build-corpus",
            "POST /bqml"
        ]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)