"""Match faith-related signals in Instagram bios and post captions.

Two kinds of signal are detected:

* ``verse``   -- a scripture reference such as "Jeremiah 29:11" or "Phil. 4:13"
* ``keyword`` -- a phrase, word, or emoji from the tiered signal table below

Every signal carries a tier. ``strong`` signals are unambiguous self-identification
("christian", "follower of Christ", a verse reference). ``weak`` signals are words
that religious people use constantly but so does everyone else -- "god", "blessed",
"faith" -- and on their own they are not evidence of anything. Tiers are reported
per match so the caller can decide where to cut.
"""

import re
import unicodedata

TIER_WEIGHTS = {"strong": 10, "medium": 4, "weak": 1}
TIER_ORDER = ["strong", "medium", "weak"]

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"))


def normalize(text):
    """Fold text to a comparable form.

    NFKC is doing real work here, not ceremony: Instagram bios are full of the
    mathematical-alphanumeric block (bold/script/fraktur "fonts") and NFKC maps
    those back to ASCII, so a bio reading "\U0001d485\U0001d489\U0001d48a\U0001d495\U0001d495\U0001d48e\U0001d486\U0001d48f" still matches "christian".
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ZERO_WIDTH)
    text = text.replace("∶", ":").replace("：", ":")  # ratio / fullwidth colon
    return re.sub(r"\s+", " ", text).strip().lower()


# --- scripture references -------------------------------------------------
#
# Aliases that collide with ordinary English words ("is" for Isaiah, "am" for
# Amos, "ac" for Acts, "kg" for Kings) are deliberately omitted -- the chapter:verse
# suffix filters most noise, but not all of it, and a false hit on a bio is worse
# than a miss on an abbreviation almost nobody uses.
_BOOKS = [
    "genesis", "gen", "exodus", "exod", "exo", "leviticus", "lev",
    "numbers", "num", "deuteronomy", "deut", "joshua", "josh",
    "judges", "judg", "ruth", "samuel", "sam", "kings", "kgs",
    "chronicles", "chron", "chr", "ezra", "nehemiah", "neh",
    "esther", "esth", "job", "psalms", "psalm", "psa", "pslm", "ps",
    "proverbs", "prov", "prv", "ecclesiastes", "eccles", "eccl", "ecc",
    "song of solomon", "song of songs", "canticles", "song",
    "isaiah", "isa", "jeremiah", "jer", "lamentations", "lam",
    "ezekiel", "ezek", "ezk", "daniel", "dan", "hosea", "hos",
    "joel", "amos", "obadiah", "obad", "jonah", "jnh", "micah", "mic",
    "nahum", "nah", "habakkuk", "hab", "zephaniah", "zeph", "zep",
    "haggai", "hag", "zechariah", "zech", "zec", "malachi", "mal",
    "matthew", "matt", "mt", "mark", "mrk", "mk", "luke", "luk", "lk",
    "john", "jhn", "joh", "jn", "acts", "romans", "rom",
    "corinthians", "cor", "galatians", "gal", "ephesians", "eph",
    "philippians", "phil", "php", "philemon", "philem", "phlm", "phm",
    "colossians", "col", "thessalonians", "thess", "thes",
    "timothy", "tim", "titus", "tit", "hebrews", "heb", "james", "jas",
    "peter", "pet", "jude", "revelations", "revelation", "rev",
]
# longest first so "song of solomon" wins over "song"
_BOOK_ALT = "|".join(re.escape(b) for b in sorted(_BOOKS, key=len, reverse=True))

VERSE_RE = re.compile(
    r"\b"
    r"(?:(?:[123]|i{1,3}|first|second|third)\s*)?"   # 1 John, II Tim, First Peter
    r"(?:" + _BOOK_ALT + r")"
    r"\.?\s*"                                        # "Phil. 4:13"
    r"\d{1,3}\s*[:.]\s*\d{1,3}"                      # chapter:verse
    r"(?:\s*[-–—,]\s*\d{1,3})*",           # ranges and lists
    re.IGNORECASE,
)

# --- keyword signals ------------------------------------------------------
# (label, tier, regex source). Word boundaries matter: r"\bgod\b" must not fire
# on "goddess" or "godmother".
_KEYWORDS = [
    ("christian",        "strong", r"\bchristian(?:ity)?\b"),
    ("follower of christ","strong", r"\bfollower of (?:christ|jesus)\b"),
    ("child of god",     "strong", r"\bchild of god\b|\bdaughter of the king\b|\bson of the king\b"),
    ("born again",       "strong", r"\bborn[ -]again\b"),
    ("saved by grace",   "strong", r"\bsaved by grace\b|\bsinner saved\b|\bredeemed\b"),
    ("god first",        "strong", r"\bgod (?:is )?first\b|\bjesus first\b|\bgod over everything\b"),
    ("christ follower",  "strong", r"\bchrist[- ]?follower\b|\bchristfollower\b"),
    ("jesus",            "medium", r"\bjesus\b|\bchrist\b|\byeshua\b|\bmessiah\b"),
    ("gospel",           "medium", r"\bgospel\b|\bscripture\b|\bbible\b|\bbiblical\b"),
    ("ministry",         "medium", r"\bministry\b|\bmissionary\b|\bmissions\b|\bdiscipleship\b|\bdisciple\b"),
    ("pastor",           "medium", r"\bpastor\b|\byouth pastor\b|\bworship leader\b|\bdeacon\b|\belder\b"),
    ("church",           "medium", r"\bchurch\b|\bcongregation\b|\bsmall group\b|\bhome group\b"),
    ("worship",          "medium", r"\bworship\b|\bhillsong\b|\belevation worship\b|\bbethel music\b"),
    ("holy spirit",      "medium", r"\bholy spirit\b|\bhis grace\b|\bhis glory\b|\bto god be the glory\b"),
    ("salvation",        "medium", r"\bsalvation\b|\bsavior\b|\bsaviour\b|\bcrucified\b|\brisen\b"),
    ("amen",             "medium", r"\bamen\b|\bhallelujah\b|\balleluia\b|\bpraise the lord\b|\bgod is good\b"),
    ("god",              "weak",   r"\bgod\b|\bthe lord\b|\blord\b"),
    ("faith",            "weak",   r"\bfaith\b|\bfaithful\b|\bprayer\b|\bpraying\b|\bprayers\b"),
    ("blessed",          "weak",   r"\bblessed\b|\bblessings\b|\bgrateful\b&\bgod\b"),
    ("cross emoji",      "weak",   r"[✝✞✟☦\U0001f54a\U0001f64f]"),
]
KEYWORD_RES = [(label, tier, re.compile(src, re.IGNORECASE)) for label, tier, src in _KEYWORDS]

# --- the church -----------------------------------------------------------
DEFAULT_CHURCH = {
    "name": "Calvary Chapel Palos Verdes",
    # Handles and hashtags are guesses at the usual naming conventions -- verify
    # the real ones against the church's own profile and edit this list.
    "patterns": [
        r"calvary\s*chapel\s*(?:of\s*)?palos\s*verdes",
        r"\bcc\s*palos\s*verdes\b",
        r"\bccpv\b",
        r"\bcalvary\s*chapel\s*pv\b",
        r"@?\s?calvarypv\b",
        r"@?\s?ccpalosverdes\b",
        r"#calvarychapelpalosverdes\b",
        r"#ccpv\b",
    ],
}
# "Calvary Chapel" with no location is a weaker, separate signal: it is a large
# denomination with hundreds of campuses, so this catches the right movement but
# not necessarily the right campus.
CALVARY_GENERIC_RE = re.compile(r"\bcalvary\s*chapel\b", re.IGNORECASE)
PALOS_VERDES_RE = re.compile(r"\bpalos\s*verdes\b|\bpve?\b", re.IGNORECASE)


def compile_church(config=None):
    cfg = config or DEFAULT_CHURCH
    return cfg.get("name", "church"), [re.compile(p, re.IGNORECASE) for p in cfg["patterns"]]


def _snippet(text, start, end, pad=40):
    left = max(0, start - pad)
    right = min(len(text), end + pad)
    return ("..." if left else "") + text[left:right].strip() + ("..." if right < len(text) else "")


def scan_text(text, church=None):
    """Return a list of match dicts for one piece of text.

    Snippets come from the normalized text, not the original -- NFKC shifts
    offsets, so quoting the raw string here would slice it in the wrong place.
    """
    norm = normalize(text)
    if not norm:
        return []

    matches = []
    seen = set()

    def add(kind, label, tier, m):
        key = (kind, label)
        if key in seen:
            return
        seen.add(key)
        matches.append({
            "kind": kind, "label": label, "tier": tier,
            "text": m.group(0).strip(), "snippet": _snippet(norm, m.start(), m.end()),
        })

    for m in VERSE_RE.finditer(norm):
        add("verse", "scripture reference", "strong", m)

    church_name, church_res = compile_church(church)
    for cre in church_res:
        m = cre.search(norm)
        if m:
            add("church", church_name, "strong", m)
            break
    else:
        m = CALVARY_GENERIC_RE.search(norm)
        if m:
            tier = "strong" if PALOS_VERDES_RE.search(norm) else "medium"
            add("church", "calvary chapel (campus unconfirmed)", tier, m)

    for label, tier, kre in KEYWORD_RES:
        m = kre.search(norm)
        if m:
            add("keyword", label, tier, m)

    return matches


def score(matches):
    return sum(TIER_WEIGHTS[m["tier"]] for m in matches)


def best_tier(matches):
    for tier in TIER_ORDER:
        if any(m["tier"] == tier for m in matches):
            return tier
    return None
