# handwriting-font

Turns photos of your handwriting into an OpenType font, then lays out card messages in it with
enough per-letter variation that a printed card reads as handwritten rather than typeset.

There are two ways to feed it. The journal pipeline mines glyphs out of ordinary notebook pages;
that is what produced the first version of the font. The template pipeline uses a printed
two-page sheet where you write each character a few times in guided boxes; it is faster and
produces a noticeably cleaner font, because fast journal writing collapses letters like n/u/m/w
and the photos only give 15-25 px per letter. If you are going to print cards with this, do the
template once.

Nothing in this folder contains handwriting. Photos, extracted glyphs, fonts and previews are
gitignored because this repository is public and a font of your hand is enough to forge notes.

## Setup

    pip install -r requirements.txt     # Python 3.10+

## Template workflow (recommended)

    python -m hwfont.template make build/template.pdf
    # print both pages, fill every box with a fine dark pen, photograph each page flat-ish
    python -m hwfont.template ingest build/glyphs_template.npy page1.jpg page2.jpg
    python -m hwfont.build_font build/glyphs_template.npy build/journal_glyphs.npy -o build/MyHand.otf

Each page has four corner markers, so photos can be taken at an angle; ingestion warps them back
onto the grid and cuts each box. The printed baseline and x-height make template glyphs exactly
placed. When a character has template samples, those replace journal samples, so passing both
files keeps journal glyphs only for anything you skipped. Write naturally rather than neatly;
the point is your everyday hand.

## Journal workflow

    export HWFONT_PHOTOS=/path/to/journal/photos
    python -m hwfont.extract index <photo_id>                   # every text line with its R-id
    python -m hwfont.extract zoom <photo_id> <R> <x0> <x1> ...  # letters with exact x-ranges
    # add labels to labels/journal_labels.txt, e.g.  e0fdfe11 R39: A c321 | D 222-273
    python -m hwfont.extract extract labels/journal_labels.txt build/journal
    # review build/journal/contact.png, delete bad labels, re-extract
    python -m hwfont.build_font build/journal/glyphs.npy -o build/MyHand.otf

How it works: the printed ruling is detected (Hough segments clustered into one curve per rule),
each rule is unwarped into a straight strip at 4x resolution, and the rule pixels are removed
except where a stroke crosses them. Rule spacing is a physical constant of the notebook, so it
normalises away photo distance and perspective. Letters are labelled by x-range or by whole
ink component (`c<x0>`), because joined handwriting cannot be segmented automatically with any
reliability. `labels/journal_labels.txt` is the reviewed label set behind the current font (about
255 samples from 8 photos); it holds only photo ids and coordinates.

## Cards and other unruled pages (auto-labelled)

Neat writing on plain card stock is the best source, and it needs no hand labelling: you type
what each line says and `hwfont.align` matches the text to the ink.

    # labels/photos.json: rotation for photos taken sideways, "source": "card"
    python -m hwfont.extract index <photo_id>              # detected lines and their R-ids
    # labels/card_text.txt:  <photo_id> R<n>: <the text of that line>
    python -m hwfont.align labels/card_text.txt labels/card_labels.txt
    python -m hwfont.extract extract labels/card_labels.txt build/cards     # review contact.png
    # bad matches go in labels/card_rejects.txt (same syntax); align leaves them out next run

Pages with fewer than 8 printed rules are treated as unruled. Lines are found by painting each
word's x-height band and smearing it sideways (words on one line fuse, neighbouring lines don't),
refined by fitting and reassigning; the baseline is shifted to the most common bottom level so
arches and crossbars don't lift it. Alignment is dynamic programming over components vs
characters with height, width, shape and word-gap costs (see `hwfont/align.py`); joined letters
are split at their thinnest connector. Vertical bands and widths are learned from the page in a
first pass. On the two cards: about 690 matched components, 774 letter labels.

`card_text.txt` and `card_labels.txt` spell out the card's text, so they are gitignored and stay
on your machine with the photos.

## Benchmark

    python -m hwfont.benchmark build/cards/glyphs.npy build/v2.otf build/MyHand.otf -o build/benchmark
    python tools/side_by_side.py <photo_id> <R> "<text>" build/cmp.png build/v2.otf build/MyHand.otf

Measures stroke weight (half-maximum width of the photographed ink), slant, ascender, capital,
digit and descender heights, letter widths and word spacing, all in x-heights, for the reference
cards and each font; plus a per-letter likeness score (centerline chamfer distance in a shared
x-height frame, so size and proportion count). Font metrics in `build_font.py` are calibrated to
the card numbers.

## Font build

`build_font` turns labelled samples into glyphs in four steps.

Size. Every glyph is scaled by its page's x-height, then pulled 80% of the way toward a per-class
height (journal pages were written at different sizes, and this hand has very tall ascender loops).

Tidy pass (`hwfont/tidy.py`, on by default). The ink is reduced to its centerline, split into
strokes, cleaned of pen-entry spurs, smoothed, and redrawn with one round pen of fixed width. The
path of the hand is kept; only the ink is replaced, so blotted loops reopen and blurry and sharp
photos produce the same weight. Joining strokes that run sideways out of a letter's body are
trimmed (otherwise "see" sets as "s-e-e"), and each glyph's lean is moved 65% of the way toward
the median slant of the whole hand. `--raw` skips all of this and traces the ink as photographed.

Sample choice. Samples come from three tiers, template > card > journal; a letter uses the
most deliberate tier that has a legible sample. Within a tier, letters cut out of joined words
rank below whole ink components, an i or j without its dot is never used when a dotted one
exists, and samples with implausible height for their letter (slivers) or a missing second mark
(" : ;) are dropped.

Legibility filter (`hwfont/legibility.py`). Fast writing collapses letters (an e becomes "<"), and
the collapsed form is often the most common one, so samples are ranked by how much their
centerline looks like the intended letter in a few reference typefaces (symmetric chamfer
distance, lower is better). Only samples scoring at or under `LEGIBLE_MAX` (4.2) are used, best
first. A letter with no passing sample is drawn instead, in the writer's slant and pen width;
Each sample must also look more like itself than like its usual confusions (n/r/u, a/u/o, e/c,
h/n/b, i/l). `FORCE_DRAWN` (m, journal only: the fast m reads as "w") and `ALWAYS_DRAWN` (n: the
card n carries an entry tick that reads as r in running text) are explicit, auditable exceptions. `legibility.json` records each used sample's score.

Gaps. Characters with no sample are filled from your own glyphs where possible (a lowered P
becomes p, your a plus a stem becomes q and g, scaled lowercase becomes a missing capital), and
otherwise drawn. `coverage.json` records the source of every character and `forms.json` the
exact label behind each form, so every glyph can be traced back to a photo and a pixel range.
`python tools/specimen.py build/MyHand.otf build/specimen.png` shows all forms with their sources.

Each character gets three forms: the best samples, then small affine perturbations. A `calt`
feature cycles forms based on the previous glyph (spaces included), so "letter" never gets two
identical t's in apps that apply contextual alternates (browsers, Pages, InDesign, Canva; Word
needs Font > Advanced > Contextual Alternates). Forms are also mapped to Private Use codepoints
listed in `<font>.pua.json`, which the card tool uses to choose forms itself.

## Card tool

    python -m hwfont.make_card_tool build/MyHand.otf build/card.html

A single offline HTML file with the font embedded. Type the message, pick card size and ink, and
it renders each letter with a varied form, slight baseline drift, rotation, size and ink-pressure
variation (the "natural wobble" slider; "Reshuffle" draws a new variation). Export as PDF through
the print dialog (vector, page size set to the card), as a 300 dpi PNG with optional transparent
background for printing onto a designed card, or as SVG. Characters not in the font fall back to
a system font and are listed under the preview.

## Files

    hwfont/journal.py         ink map, rule detection, strip unwarping, rule removal, components
    hwfont/extract.py         index/zoom views for labelling, label parser, glyph extraction
    hwfont/build_font.py      normalisation, sample selection, tracing, fallbacks, alternates, OpenType
    hwfont/tidy.py            centerline extraction, spur pruning, connector trimming, re-stroking
    hwfont/legibility.py      reads-as-letter score against reference typefaces
    hwfont/align.py           auto-labelling from a transcription (unruled pages, cards)
    hwfont/benchmark.py       style metrics and per-letter likeness against reference handwriting
    hwfont/template.py        template PDF and photo ingestion
    hwfont/make_card_tool.py  bakes a font into hwfont/card_tool.html
    tests/test_template_roundtrip.py   fills a template with a font, fakes a skewed photo, ingests
    tools/preview.js, tools/test_card.js   Playwright renders for visual checks
    tools/specimen.py         every form of every character with the sample it came from
    tools/side_by_side.py     a real line from a photo above the same text set in each font
