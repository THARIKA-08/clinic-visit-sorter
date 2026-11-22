"""Clinic Visit Sorter - console tool.

Reads visits.csv (id,time,note), asks an LLM to label each note, applies
Python safety rules, and writes nurse_queue.csv + summary.txt.

The tool NEVER diagnoses, NEVER lowers a priority, NEVER treats unknown as
routine, and NEVER removes a student from the queue.
"""
import csv
import html
import json
import os
import re
import sys
import time

# ----------------------------------------------------------------- settings
PROMPT_FILE = "system_prompt.txt"
MAX_NOTE_CHARS = 500            # only this much goes to the model
MAX_SUMMARY_WORDS = 12
MAX_ATTEMPTS = 2                # first try + one retry
NEUTRAL_SUMMARY = "Symptoms reported, see original note"
# Free-tier APIs limit calls per minute, so we pause a little (seconds).
RETRY_WAIT = float(os.environ.get("CLINIC_RETRY_WAIT", "6"))   # before the one retry
NOTE_DELAY = float(os.environ.get("CLINIC_DELAY", "5"))        # between notes

CATEGORIES = ["injury", "fever_cold", "stomach", "pain", "allergy", "other"]
PRIORITIES = ["see_now", "today", "routine"]
PRIORITY_RANK = {"see_now": 0, "today": 1, "routine": 2}   # lower = more urgent

# Python forces see_now if any of these phrases appears in the RAW note.
RED_FLAGS = [
    # chest / breathing
    "chest pain", "chest tightness", "difficulty breathing", "difficulty in breathing",
    "can't breathe", "cant breathe", "cannot breathe", "breathless", "short of breath",
    "breathing trouble", "trouble breathing", "breathing problem", "breathing difficulty",
    "hard to breathe", "struggling to breathe",
    "நெஞ்சு வலி", "மூச்சு திணறல்", "nenju vali", "nenju valikuthu", "moochu",
    # fainting
    "fainting", "fainted", "faint", "passed out", "unconscious", "மயக்கம்", "mayakkam",
    # bleeding
    "heavy bleeding", "bleeding heavily", "bleeding a lot", "lot of blood", "blood loss",
    "won't stop bleeding", "wont stop bleeding", "vomiting blood",
    # swelling of lip / face / tongue
    "lip swelling", "face swelling", "tongue swelling", "swollen lip", "swollen face",
    "swollen tongue", "lips are swollen", "face is swollen", "lip swollen", "face swollen",
    "lips swollen", "swelling of lip", "swelling of face", "swelling in lip",
    "swelling in face", "swelling of the lip", "swelling of the face",
    "swelling of the tongue", "udhadu veekkam",
    "mugam veekkam",
    # cannot stand
    "can't stand", "cant stand", "cannot stand", "unable to stand", "nikka mudiyala",
]

# Words that must never appear in a summary (no diagnosis, no guessing).
DISEASE_WORDS = [
    "dengue", "malaria", "typhoid", "covid", "corona", "flu", "influenza", "pneumonia",
    "asthma", "migraine", "appendicitis", "gastritis", "gastroenteritis", "ulcer",
    "infection", "infected", "fracture", "fractured", "sprain", "sprained",
    "allergy", "allergic", "conjunctivitis", "pink eye", "food poisoning", "diabetes",
    "anemia", "anaemia", "tonsillitis", "bronchitis", "jaundice", "chickenpox",
    "measles", "heart attack", "stroke", "anxiety", "depression", "viral", "bacterial",
    "sinusitis", "dehydration", "concussion",
]
GUESS_WORDS = ["possible", "possibly", "likely", "probably", "maybe", "suspected",
               "might be", "could be", "seems like", "looks like"]
_BANNED_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in DISEASE_WORDS + GUESS_WORDS) + r")\b",
    re.IGNORECASE,
)

TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*([AaPp][Mm])?\s*$")


# ------------------------------------------------------------------ loading
def parse_time(text):
    """Return minutes since midnight, or None if unreadable. Accepts 12h and 24h."""
    m = TIME_RE.match(text or "")
    if not m:
        return None
    hour, minute, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
    if minute > 59:
        return None
    if ampm:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if ampm.lower() == "pm" else 0)
    elif hour > 23:
        return None
    return hour * 60 + minute


def load_visits(path):
    """Read visits.csv into a list of dicts. Never drops a row."""
    visits = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for index, row in enumerate(csv.DictReader(f)):
            note = html.unescape((row.get("note") or "").strip())   # &#39; -> '
            visits.append({
                "id": (row.get("id") or f"ROW{index + 1}").strip(),
                "time": (row.get("time") or "").strip(),
                "minutes": parse_time(row.get("time")),
                "note": note,
                "booked_emergency": (row.get("booking_priority") or "").strip().lower() == "emergency",
                "order": index,
            })
    return visits


# --------------------------------------------------------------- LLM access
_client = None
_system_prompt = None


def get_provider():
    """Pick the LLM provider from whichever API key is set in the environment."""
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise RuntimeError("No API key set (GEMINI_API_KEY or ANTHROPIC_API_KEY)")


def get_system_prompt():
    global _system_prompt
    if _system_prompt is None:
        with open(PROMPT_FILE, encoding="utf-8") as f:
            _system_prompt = f.read()
    return _system_prompt


def call_llm(note):
    """One API call. Returns the raw reply text. May raise on any API problem.
    The key is read from an environment variable - never written in the code."""
    global _client
    user_message = f"Student note:\n{note[:MAX_NOTE_CHARS]}"

    if get_provider() == "gemini":
        from google import genai
        from google.genai import types
        if _client is None:
            _client = genai.Client()          # reads GEMINI_API_KEY / GOOGLE_API_KEY
        response = _client.models.generate_content(
            model=os.environ.get("CLINIC_MODEL", "gemini-3.8-flash"),
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=get_system_prompt(),
                temperature=0,
                max_output_tokens=1000,
                response_mime_type="application/json",
            ),
        )
        return response.text or ""

    import anthropic                          # reads ANTHROPIC_API_KEY
    if _client is None:
        _client = anthropic.Anthropic()
    response = _client.messages.create(
        model=os.environ.get("CLINIC_MODEL", "claude-haiku-5-5"),
        max_tokens=200,
        temperature=0,
        system=get_system_prompt(),
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(b.text for b in response.content if getattr(b, "text", None))


# --------------------------------------------------------------- validation
def is_valid(reply_text):
    """Return the parsed dict if the reply is valid, otherwise None."""
    try:
        text = reply_text.strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        data = json.loads(text)
    except (AttributeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("category") not in CATEGORIES:
        return None
    if data.get("priority") not in PRIORITIES:
        return None
    if not isinstance(data.get("summary"), str) or not data["summary"].strip():
        return None
    return {"category": data["category"], "priority": data["priority"],
            "summary": data["summary"].strip()}


def classify(note):
    """Ask the LLM up to MAX_ATTEMPTS times. Fallback: today / other (never routine)."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = is_valid(call_llm(note))
        except Exception as error:            # API down, timeout, bad key...
            print(f"   attempt {attempt} failed: {type(error).__name__}: {str(error)[:200]}", file=sys.stderr)
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_WAIT)
            continue
        if result:
            result["from_llm"] = True
            return result
        print(f"   attempt {attempt}: invalid reply", file=sys.stderr)
    return {"category": "other", "priority": "today",
            "summary": NEUTRAL_SUMMARY, "from_llm": False}


# ------------------------------------------------------------- Python rules
def _normalise(text):
    return text.lower().replace("\u2019", "'").replace("\u2018", "'")


def has_red_flag(note):
    text = _normalise(note)
    return any(flag.lower() in text for flag in RED_FLAGS)


def raise_priority(priority, note, booked_emergency=False):
    """Priority can only go UP. A red flag (or Emergency booking) forces see_now."""
    if has_red_flag(note) or booked_emergency:
        return "see_now"
    return priority


def remove_diagnosis(summary):
    """Replace diagnostic / guessing summaries; cut to 12 words."""
    if not summary or _BANNED_RE.search(summary):
        return NEUTRAL_SUMMARY
    words = summary.split()
    return " ".join(words[:MAX_SUMMARY_WORDS])


def process_visit(visit):
    """Full pipeline for one student. Never raises, never loses the student."""
    note = visit["note"]
    if not note:
        result = {"category": "other", "priority": "today", "summary": "No note given"}
    else:
        result = classify(note)
    priority = raise_priority(result["priority"], note, visit.get("booked_emergency", False))
    # safety net: the final priority can never be lower than what the LLM said
    if PRIORITY_RANK[priority] > PRIORITY_RANK[result["priority"]]:
        priority = result["priority"]
    return {**visit, "category": result["category"], "priority": priority,
            "summary": remove_diagnosis(result["summary"])}


# ------------------------------------------------------------------ reports
def sort_queue(rows):
    """see_now > today > routine, then arrival time; unreadable time goes last."""
    return sorted(rows, key=lambda r: (
        PRIORITY_RANK[r["priority"]],
        r["minutes"] is None,
        r["minutes"] if r["minutes"] is not None else 0,
        r["order"],
    ))


def hour_label(minutes):
    h = minutes // 60
    return f"{(h % 12) or 12:02d}:00 {'AM' if h < 12 else 'PM'}"


def write_reports(rows, queue_path="nurse_queue.csv", summary_path="summary.txt"):
    ordered = sort_queue(rows)

    with open(queue_path, "w", newline="", encoding="utf-8") as f:       # nurse only
        writer = csv.writer(f)
        writer.writerow(["id", "time", "priority", "category", "summary", "note"])
        for r in ordered:
            writer.writerow([r["id"], r["time"], r["priority"], r["category"],
                             r["summary"], r["note"]])

    by_priority = {p: 0 for p in PRIORITIES}
    by_category = {c: 0 for c in CATEGORIES}
    by_hour, unknown_time = {}, 0
    for r in rows:
        by_priority[r["priority"]] += 1
        by_category[r["category"]] += 1
        if r["minutes"] is None:
            unknown_time += 1
        else:
            by_hour[r["minutes"] // 60] = by_hour.get(r["minutes"] // 60, 0) + 1

    hour_parts = []
    if by_hour:
        for h in range(min(by_hour), max(by_hour) + 1):                  # gaps show as 0
            hour_parts.append(f"{hour_label(h * 60)} {by_hour.get(h, 0)}")
    if unknown_time:
        hour_parts.append(f"unknown {unknown_time}")

    lines = [                                                            # counts only
        f"Total visits: {len(rows)}",
        "By priority:  " + " | ".join(f"{p} {n}" for p, n in by_priority.items()),
        "By category:  " + " | ".join(f"{c} {n}" for c, n in by_category.items()),
        "By hour:      " + " | ".join(hour_parts),
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return lines


# --------------------------------------------------------------------- main
def main(input_path="visits.csv"):
    try:
        visits = load_visits(input_path)
    except (OSError, csv.Error) as error:
        sys.exit(f"Could not read {input_path}: {error}")

    try:
        print(f"Using LLM provider: {get_provider()}")
    except RuntimeError as error:
        print(f"WARNING: {error}. Every note will use the fallback priority 'today'.\n"
              "Set the key in this terminal and run again for real LLM results.\n")

    start = time.time()
    rows = []
    for i, visit in enumerate(visits, 1):
        print(f"[{i}/{len(visits)}] {visit['id']}")
        try:
            rows.append(process_visit(visit))
            if i < len(visits) and visit["note"] and NOTE_DELAY > 0:
                time.sleep(NOTE_DELAY)
        except Exception as error:            # one bad note must never stop the batch
            print(f"   unexpected error ({error}); using fallback", file=sys.stderr)
            rows.append({**visit, "category": "other",
                         "priority": raise_priority("today", visit["note"], visit["booked_emergency"]),
                         "summary": NEUTRAL_SUMMARY})

    # Queue safeguard: AI triage only changes ORDER; every student must still be in the queue.
    assert len(rows) == len(visits), "a student was lost from the queue"
    assert {r["id"] for r in rows} == {v["id"] for v in visits}, "queue ids changed"

    for line in write_reports(rows):
        print(line)
    elapsed = time.time() - start
    print(f"\nDone: {len(rows)} students, {elapsed:.1f}s total "
          f"({elapsed / max(len(rows), 1):.1f}s per note).")
    print("Wrote nurse_queue.csv (nurse only) and summary.txt (in-charge).")
    print("Human checkpoint: the nurse sees every patient and makes every medical decision.")
    print("This tool does not diagnose, give advice, or remove anyone from the queue.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "visits.csv")
