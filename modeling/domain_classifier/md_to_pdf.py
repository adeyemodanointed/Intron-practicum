"""Convert a simple Markdown file (headings, paragraphs, lists, tables, bold/italic/links) to a styled PDF.
    python md_to_pdf.py KEY_PAPERS_FOR_PRESENTATION.md KEY_PAPERS_FOR_PRESENTATION.pdf
"""
import re, sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import CondPageBreak, HRFlowable, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

F = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("Body", F + "calibri.ttf"))
pdfmetrics.registerFont(TTFont("Body-Bold", F + "calibrib.ttf"))
pdfmetrics.registerFont(TTFont("Body-Italic", F + "calibrii.ttf"))
pdfmetrics.registerFont(TTFont("Body-BoldItalic", F + "calibrib.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic")

INK, MUTED, ACCENT, RULE, HEAD_BG = colors.HexColor("#1f2328"), colors.HexColor("#57606a"), colors.HexColor("#1c5cab"), colors.HexColor("#d0d7de"), colors.HexColor("#eef3fa")
S = {
    "h1": ParagraphStyle("h1", keepWithNext=1, fontName="Body-Bold", fontSize=20, leading=24, textColor=INK, spaceAfter=8),
    "h2": ParagraphStyle("h2", keepWithNext=1, fontName="Body-Bold", fontSize=15, leading=19, textColor=ACCENT, spaceBefore=10, spaceAfter=6),
    "h3": ParagraphStyle("h3", keepWithNext=1, fontName="Body-Bold", fontSize=12, leading=15, textColor=INK, spaceBefore=8, spaceAfter=4),
    "p": ParagraphStyle("p", fontName="Body", fontSize=10.5, leading=14, textColor=INK, spaceAfter=5, alignment=TA_LEFT),
    "li": ParagraphStyle("li", fontName="Body", fontSize=10.5, leading=14, textColor=INK),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.8, leading=11, textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName="Body-Bold", fontSize=8.8, leading=11, textColor=INK),
}


def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: f'<link href="{m.group(2)}" color="#1c5cab"><u>{m.group(1)}</u></link>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"`([^`]+)`", r'<font name="Courier" size="9">\1</font>', t)
    return t


def table(lines, width):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    head, body = rows[0], [r for r in rows[2:]]
    ncol = len(head)
    data = [[Paragraph(inline(c), S["cellh"]) for c in head]] + [[Paragraph(inline(c), S["cell"]) for c in (r + [""] * ncol)[:ncol]] for r in body]
    from reportlab.pdfbase.pdfmetrics import stringWidth
    plain = lambda c: re.sub(r"\[([^\]]+)\]\([^)]+\)", lambda m: m.group(1), c).replace("**", "").replace("`", "").replace("*", "")
    cells = [r for r in rows if not set("".join(r)) <= set("-: ")]
    need = [max(stringWidth(wd, "Body-Bold", 8.8) for r in cells if i < len(r) for wd in (plain(r[i]).split() or [""])) + 10 for i in range(ncol)]
    want = [max(stringWidth(plain(r[i]), "Body", 8.8) for r in cells if i < len(r)) + 10 for i in range(ncol)]
    if sum(want) <= width:
        cw = [x * width / sum(want) for x in want]
    else:
        extra = width - sum(need)
        grow = [max(0, w - n) for w, n in zip(want, need)]
        cw = [n + (extra * g / sum(grow) if sum(grow) else 0) for n, g in zip(need, grow)] if extra > 0 else [n * width / sum(need) for n in need]
    t = Table(data, colWidths=cw, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
                           ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return t


def lists(items):
    """items: (indent, kind, text). Builds nested ListFlowables."""
    def build(i, level):
        out, kind = [], items[i][1]
        while i < len(items) and items[i][0] >= level:
            ind, k, txt = items[i]
            if ind > level:
                sub, i = build(i, ind)
                out[-1].append(sub)
                continue
            out.append([Paragraph(inline(txt), S["li"])]); i += 1
        flow = ListFlowable([ListItem(x if len(x) > 1 else x[0], leftIndent=14) for x in out],
                            bulletType="1" if kind == "ol" else "bullet", start="1" if kind == "ol" else None,
                            bulletFontName="Body", bulletFontSize=9 if kind == "ul" else 10, leftIndent=14,
                            bulletColor=ACCENT if kind == "ul" else INK)
        return flow, i
    return build(0, items[0][0])[0]


def convert(src, dst):
    lines = open(src, encoding="utf-8").read().splitlines()
    doc = SimpleDocTemplate(dst, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title=lines[0].lstrip("# ").strip(), author="Michael Sangwa")
    width = A4[0] - 36 * mm
    story, i, para = [], 0, []

    def flush():
        if para:
            story.append(Paragraph(inline(" ".join(para)), S["p"])); para.clear()
    while i < len(lines):
        l = lines[i]
        if not l.strip():
            flush(); i += 1; continue
        if l.startswith("#"):
            flush(); n = len(l) - len(l.lstrip("#"))
            story.append(Paragraph(inline(l[n:].strip()), S[f"h{min(n, 3)}"]))
            if n == 1:
                story.append(HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceAfter=6))
            i += 1; continue
        if l.strip() == "---":
            flush(); story.append(HRFlowable(width="100%", thickness=0.5, color=RULE, spaceBefore=6, spaceAfter=6)); story.append(CondPageBreak(70 * mm)); i += 1; continue
        if l.lstrip().startswith("|"):
            flush(); block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i]); i += 1
            story += [Spacer(1, 3), table(block, width), Spacer(1, 6)]; continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", l)
        if m:
            flush(); items = []
            while i < len(lines):
                m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", lines[i])
                if not m:
                    if lines[i].strip() and lines[i].startswith("   ") and items:   # continuation line
                        items[-1] = (items[-1][0], items[-1][1], items[-1][2] + " " + lines[i].strip()); i += 1; continue
                    break
                items.append((len(m.group(1)) // 2, "ol" if m.group(2)[0].isdigit() else "ul", m.group(3))); i += 1
            story += [lists(items), Spacer(1, 5)]; continue
        para.append(l.strip()); i += 1
    flush()

    def footer(c, d):
        c.saveState(); c.setFont("Body", 8.5); c.setFillColor(MUTED)
        c.drawString(18 * mm, 9 * mm, "Multilingual domain classifier · CMU-Africa MSIT practicum with Intron")
        c.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {d.page}"); c.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
