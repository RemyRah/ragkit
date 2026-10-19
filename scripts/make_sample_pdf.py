#!/usr/bin/env python3
"""Regenerate sample_docs/shop/brewing-guide.pdf with only the standard library.

A real (if minimal) PDF: one Helvetica page per section group, text drawn with
plain PDF operators, and a correct xref table so any reader opens it.

    python scripts/make_sample_pdf.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "sample_docs" / "shop" / "brewing-guide.pdf"

SECTIONS = [
    ("Brewing Guide", [
        "Ember & Oak Coffee Roasters is a fictional shop used as ragkit demo data.",
    ]),
    ("Pour-over", [
        "Use 22 grams of medium-fine coffee for 350 grams of water at 94C (200F). "
        "Bloom with 50 grams of water for 40 seconds, then pour slowly. Total brew time 3 minutes.",
    ]),
    ("French press", [
        "Use 30 grams of coarse coffee for 500 grams of water just off the boil. "
        "Steep 4 minutes, stir, skim the crust, then press gently and serve right away.",
    ]),
    ("Espresso", [
        "Dose 18 grams of fine coffee and pull about 36 grams of espresso in 25 to 30 seconds. "
        "Midnight Espresso is roasted for this; rest the beans 7 days after roasting first.",
    ]),
    ("Cold brew", [
        "Mix 100 grams of coarse coffee with 1 litre of cold water and steep 16 hours in the fridge. "
        "Strain and dilute 1 to 1 with water or milk. Sunrise Blend is our favourite for cold brew.",
    ]),
]


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _page_stream() -> bytes:
    ops = ["BT", "/F1 11 Tf", "14 TL", "72 740 Td"]
    for i, (heading, paras) in enumerate(SECTIONS):
        size = 18 if i == 0 else 13
        ops += [f"/F1 {size} Tf", f"({_escape(heading)}) Tj", "T*", "/F1 11 Tf"]
        for para in paras:
            for line in textwrap.wrap(para, 85):
                ops += [f"({_escape(line)}) Tj", "T*"]
        ops.append("T*")
    ops.append("ET")
    return "\n".join(ops).encode("latin-1")


def build_pdf() -> bytes:
    stream = _page_stream()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


if __name__ == "__main__":
    OUT.write_bytes(build_pdf())
    print(f"wrote {OUT}")
