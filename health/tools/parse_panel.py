#!/usr/bin/env python3
"""Parse a lab-panel PDF into structured, triaged markers.

Handles the "Ranges Core Health Test" layout (The Health Institute / SiPhox /
Function-style panels). A marker is a block:

    Marker Name (unit) optimal: LO - HI      <- header: name, optional unit, first tier
    good: LO - HI                            <- zero or more further tiers
    fair: LO - HI
    VALUE                                    <- value on its own line...

...or, when the marker has only one tier, the value trails on the header line:

    Ferritin (ng/mL) optimal: 70 - 90 21.1

Usage:
    python3 health/tools/parse_panel.py <panel.pdf> [-o out.json]

Writes JSON (stdout or -o) and a triage table to stderr.
Requires: pip install pypdf
"""
import argparse, json, re, sys

TIER = re.compile(r"\b(optimal|good|fair|standard|normal)\s*:\s*"
                  r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", re.I)
LEADING_TIER = re.compile(r"^\s*(optimal|good|fair|standard|normal)\s*:", re.I)
UNIT = re.compile(r"\(([^)]*)\)\s*$")
NUM = re.compile(r"-?\d+(?:\.\d+)?")
STOP = re.compile(r"^\s*(insights?|recommendations?|disclaimer)\b", re.I)
SKIP = re.compile(r"^(=====|exported:|female$|male$)", re.I)


def extract_text(path):
    try:
        from pypdf import PdfReader
    except ImportError:
        sys.exit("pypdf required:  pip install pypdf")
    return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)


def header_value(head, name):
    """A single-tier marker trails its value on the header line: 'Ferritin (ng/mL) optimal: 70 - 90 21.1'"""
    return NUM.findall(TIER.sub(" ", head).replace(name, " "))


def triage(value, tiers):
    """GREEN at optimal; AMBER inside a wider tier; RED outside every tier."""
    if not tiers:
        return "CONTEXT"
    opt = tiers.get("optimal")
    if opt and opt[0] <= value <= opt[1]:
        return "GREEN"
    if any(lo <= value <= hi for lo, hi in tiers.values()):
        return "AMBER"
    ceiling = max(hi for _, hi in tiers.values())
    return "RED_HIGH" if value > ceiling else "RED_LOW"


def parse(text):
    markers, section, block = [], None, []

    def flush():
        if not block:
            return
        head, rest = block[0], block[1:]
        tiers = {t.lower(): [float(lo), float(hi)] for t, lo, hi in TIER.findall(" ".join(block))}

        # Name and unit come from the header, left of its first tier.
        name = head[:TIER.search(head).start()].strip() if TIER.search(head) else head.strip()
        unit = None
        if (m := UNIT.search(name)):
            unit, name = m.group(1).strip(), name[:m.start()].strip()
        name = name.strip(" :-")

        # The value is whatever number survives stripping every "tier: lo - hi".
        tail = TIER.sub(" ", head) + " " + " ".join(l for l in rest if not LEADING_TIER.match(l))
        nums = NUM.findall(TIER.sub(" ", tail).replace(name, " "))
        if name and nums:
            v = float(nums[-1])
            markers.append({"section": section, "name": name, "unit": unit,
                            "value": v, "ranges": tiers, "status": triage(v, tiers)})
        block.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line or SKIP.match(line):
            continue
        if STOP.match(line):
            flush(); break
        if LEADING_TIER.match(line):                     # continuation tier
            block.append(line)
        elif TIER.search(line):                          # new marker header
            flush(); block.append(line)
            name = line[:TIER.search(line).start()].strip()
            if (m := UNIT.search(name)):
                name = name[:m.start()].strip()
            if header_value(line, name.strip(" :-")):    # value trailed on the header
                flush()
        elif NUM.fullmatch(line):
            if block:
                block.append(line); flush()              # value on its own line
            # else: a bare number with no open block is a page number -> drop
        elif not NUM.search(line) and len(line) < 45:    # section heading
            flush(); section = line
    flush()
    return markers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf"); ap.add_argument("-o", "--out")
    a = ap.parse_args()
    markers = parse(extract_text(a.pdf))
    blob = json.dumps(markers, indent=2)
    open(a.out, "w").write(blob) if a.out else print(blob)

    rank = {"RED_HIGH": 0, "RED_LOW": 0, "AMBER": 1, "CONTEXT": 2, "GREEN": 3}
    print(f"\n{len(markers)} markers parsed\n", file=sys.stderr)
    for m in sorted(markers, key=lambda m: (rank.get(m["status"], 9), m["name"])):
        opt = m["ranges"].get("optimal")
        rng = f"optimal {opt[0]}-{opt[1]}" if opt else \
              ", ".join(f"{k} {v[0]}-{v[1]}" for k, v in m["ranges"].items())
        print(f"  {m['status']:<9} {m['name'][:42]:<43} {m['value']:>8}   {rng}", file=sys.stderr)


if __name__ == "__main__":
    main()
