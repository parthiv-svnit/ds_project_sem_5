"""
Convert presentation/speaker_script.txt → presentation/speaker_script.docx
Uses python-docx. Run from the project root:
    python src/convert_script_to_docx.py
"""
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import re

SRC = Path("presentation/speaker_script.txt")
DST = Path("presentation/speaker_script.docx")

# ── colour palette ──────────────────────────────────────────────────────────
BLUE   = RGBColor(0x00, 0x52, 0x9B)   # Speaker 1 (Divij)
GREEN  = RGBColor(0x1A, 0x7A, 0x1A)   # Speaker 2 (Parthiv)
PURPLE = RGBColor(0x6A, 0x0D, 0x83)   # Q&A header
GRAY   = RGBColor(0x88, 0x88, 0x88)   # separator / meta lines
DARK   = RGBColor(0x1A, 0x1A, 0x1A)   # body text


def _run(para, text, bold=False, italic=False, size_pt=None, colour=None):
    run = para.add_run(text)
    run.bold = bold
    run.italic = italic
    if size_pt:
        run.font.size = Pt(size_pt)
    if colour:
        run.font.color.rgb = colour
    return run


def build_docx(src: Path, dst: Path):
    doc = Document()

    # ── default body style ──────────────────────────────────────────────────
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # ── narrow margins ──────────────────────────────────────────────────────
    for section in doc.sections:
        section.top_margin    = Inches(0.9)
        section.bottom_margin = Inches(0.9)
        section.left_margin   = Inches(1.0)
        section.right_margin  = Inches(1.0)

    lines = src.read_text(encoding="utf-8").splitlines()

    in_qa = False
    prev_blank = False

    for raw in lines:
        line = raw.rstrip()

        # ── Top banner (===…===) ────────────────────────────────────────────
        if re.match(r"^={10,}$", line):
            p = doc.add_paragraph()
            _run(p, line, colour=GRAY, size_pt=9)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after  = Pt(0)
            continue

        # ── Banner content (first 8 lines: title block) ─────────────────────
        # Let them fall through to body; we identify them by position later.

        # ── Slide separator (---…---) ───────────────────────────────────────
        if re.match(r"^-{10,}$", line):
            p = doc.add_paragraph()
            _run(p, "─" * 72, colour=GRAY, size_pt=8)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after  = Pt(2)
            continue

        # ── Q&A section header ──────────────────────────────────────────────
        if re.match(r"^={3,}\s*PROFESSOR", line, re.I):
            in_qa = True
            p = doc.add_heading("", level=1)
            p.clear()
            _run(p, line.strip("= ").strip(), bold=True, size_pt=14, colour=PURPLE)
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after  = Pt(6)
            continue

        # ── SLIDE N – TITLE heading ─────────────────────────────────────────
        m_slide = re.match(r"^(SLIDE\s+\d+\s*[-–]\s*.+)$", line)
        if m_slide:
            p = doc.add_heading("", level=1)
            p.clear()
            _run(p, m_slide.group(1), bold=True, size_pt=13, colour=DARK)
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after  = Pt(2)
            continue

        # ── Speaker / Target Time line ──────────────────────────────────────
        m_speaker = re.match(r"^(Speaker:\s*)(.*?)\s*\|\s*(Target Time:.*)$", line)
        if m_speaker:
            p = doc.add_paragraph()
            _run(p, m_speaker.group(1), italic=True, size_pt=10, colour=GRAY)
            speaker_tag = m_speaker.group(2)
            colour = BLUE if "DIVIJ" in speaker_tag.upper() else GREEN
            _run(p, speaker_tag, bold=True, italic=True, size_pt=10, colour=colour)
            _run(p, "  |  ", size_pt=10, colour=GRAY)
            _run(p, m_speaker.group(3), italic=True, size_pt=10, colour=GRAY)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after  = Pt(6)
            continue

        # ── Title-block meta lines (lines 2-7 of the file) ─────────────────
        if re.match(r"^(Hybrid Semantic|Presenters:|Supervisor|Department|Total Planned)", line):
            p = doc.add_paragraph()
            _run(p, line, bold=("Hybrid Semantic" in line), size_pt=11 if "Hybrid" in line else 10,
                 colour=DARK if "Hybrid" in line else GRAY)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after  = Pt(1)
            continue

        # ── Q&A numbered questions ──────────────────────────────────────────
        if in_qa:
            m_q = re.match(r"^(\d+\.\s+)(Q:|A:|FOLLOW-UP|MODEL ANSWER|.*?(\[.*?answers\]))", line)
            if m_q or re.match(r"^\d+\.", line):
                # question number line
                p = doc.add_paragraph()
                _run(p, line, bold=True, size_pt=11, colour=DARK)
                p.paragraph_format.space_before = Pt(8)
                p.paragraph_format.space_after  = Pt(2)
                continue
            # Q: / A: prefixes inside QA block
            m_qa_line = re.match(r"^(Q:|A:|FOLLOW-UP Q:|MODEL ANSWER:)(.*)", line)
            if m_qa_line:
                p = doc.add_paragraph()
                _run(p, m_qa_line.group(1), bold=True, size_pt=10,
                     colour=BLUE if m_qa_line.group(1).startswith("Q") else GREEN)
                _run(p, m_qa_line.group(2), size_pt=10, colour=DARK)
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after  = Pt(1)
                continue

        # ── Blank lines ─────────────────────────────────────────────────────
        if line == "":
            if not prev_blank:
                doc.add_paragraph().paragraph_format.space_after = Pt(4)
            prev_blank = True
            continue

        prev_blank = False

        # ── Regular body paragraph ──────────────────────────────────────────
        p = doc.add_paragraph()
        _run(p, line, size_pt=11, colour=DARK)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after  = Pt(4)
        p.paragraph_format.first_line_indent = Pt(0)

    dst.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(dst))
    print(f"Saved -> {dst}  ({dst.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build_docx(SRC, DST)
