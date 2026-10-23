"""Layout check for the submission PDF: every main-text page must end at the end of a sentence.

    python check_page_breaks.py [main.pdf]

For each page boundary of the main text (up to the page holding the acknowledgements), it prints the last line of
the page and the first line of the next, and marks a boundary SPLIT when the page ends mid-sentence. Floats are
pinned with [H] and pages may run short (\\raggedbottom), so a split is fixed by starting the crossing sentence on
the next page (a paragraph break plus \\newpage before it). Exits 1 if any split is found. Needs PyMuPDF.
"""
import re
import sys

import pymupdf


def body_lines(page):
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text and not re.fullmatch(r"\d+", text) and text != "Preprint.":
                out.append((line["bbox"][1], text))
    return sorted(out)


def main(path="main.pdf"):
    doc = pymupdf.open(path)
    last_main = next(i for i, p in enumerate(doc) if "Acknowledgments" in p.get_text() or "Acknowledgements" in p.get_text())
    splits = 0
    for i in range(last_main):
        last, first = body_lines(doc[i])[-1][1], body_lines(doc[i + 1])[0][1]
        ok = bool(re.search(r"[.:]\s*$", last))
        splits += not ok
        print(f"p{i + 1}->p{i + 2} {'ok   ' if ok else 'SPLIT'} ...{last[-50:]!r} | {first[:40]!r}")
    print(f"main text ends on page {last_main + 1}; {splits} mid-sentence page break(s)")
    return 1 if splits else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
