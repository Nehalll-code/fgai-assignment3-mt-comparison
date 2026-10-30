"""Build the LMS submission document: a .docx cover sheet and its PDF.

The assignment asks for a Word document named
``YourName_RegNum_FGAI_Assignment3.docx`` carrying the student details, the
assignment and system titles, the dataset name and link, and the GitHub
repository link - then the same document converted to PDF for upload.

Run:

    python scripts/make_submission_doc.py

Conversion to PDF is attempted with Microsoft Word, then LibreOffice, then
docx2pdf.  If none is available the .docx is still written and the script says
so, so the document can be converted by hand (File -> Save As -> PDF).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

STUDENT_NAME = "Nehal Verma"
REG_NO = "230953144"
COURSE = "Foundations of Generative AI (ICT4403)"
ASSIGNMENT_TITLE = "Assignment 3 - Machine Translation with Encoder-Decoder and Decoder-Only Generative Models"
SYSTEM_TITLE = ("PolyGlot-4: A Comparative English-to-Hindi Machine Translation "
                "System across Encoder-Decoder and Decoder-Only Architectures")
DATASET_NAME = "OPUS-100 (English-Hindi configuration)"
DATASET_HF_ID = "Helsinki-NLP/opus-100, config en-hi"
DATASET_LINK = "https://huggingface.co/datasets/Helsinki-NLP/opus-100"
DATASET_ORIGIN_LINK = "https://opus.nlpl.eu/"
REPO_LINK = "https://github.com/Nehalll-code/fgai-assignment3-mt-comparison"

DOC_STEM = f"{STUDENT_NAME.replace(' ', '')}_{REG_NO}_FGAI_Assignment3"

MODELS = [
    ("Basic Transformer (encoder-decoder)",
     "Trained from scratch; 3+3 layers, d_model 256, joint byte-level BPE (16k)"),
    ("mT5-small (encoder-decoder)",
     "google/mt5-small, fully fine-tuned with a translation task prefix"),
    ("Llama-3.2-1B-Instruct (decoder-only)",
     "LoRA fine-tuned (r=16), plus a 3-shot prompted contrast"),
    ("Gemma-2-2B-IT (decoder-only)",
     "LoRA fine-tuned (r=16), plus a 3-shot prompted contrast"),
]


def _results_summary() -> list[tuple[str, str]]:
    """Best-model line taken from results/metrics.csv when a run exists."""
    import csv

    metrics = ROOT / "results" / "metrics.csv"
    info_path = ROOT / "results" / "run_info.json"
    if not metrics.exists():
        return []
    with metrics.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    profile = "unknown"
    if info_path.exists():
        profile = json.loads(info_path.read_text(encoding="utf-8")).get("profile", "?")
    if profile != "full":
        return []
    best = max(rows, key=lambda r: float(r["chrf"]))
    return [
        ("Best-performing model",
         f"{best['model']} - BLEU {best['bleu']}, chrF++ {best['chrf']} "
         f"on 500 held-out test sentences"),
    ]


def build_docx(path: Path) -> Path:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt, RGBColor

    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    heading = doc.add_paragraph()
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = heading.add_run(COURSE)
    run.bold = True
    run.font.size = Pt(16)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run(ASSIGNMENT_TITLE)
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    doc.add_paragraph()

    rows = [
        ("Student name", STUDENT_NAME),
        ("Registration number", REG_NO),
        ("Course", COURSE),
        ("Assignment title", ASSIGNMENT_TITLE),
        ("Title of the developed system", SYSTEM_TITLE),
        ("Dataset name", f"{DATASET_NAME} - Hugging Face id: {DATASET_HF_ID}"),
        ("Dataset link", DATASET_LINK),
        ("Original corpus", DATASET_ORIGIN_LINK),
        ("GitHub repository link", REPO_LINK),
    ] + _results_summary()

    table = doc.add_table(rows=0, cols=2)
    table.style = "Table Grid"
    for label, value in rows:
        cells = table.add_row().cells
        p = cells[0].paragraphs[0]
        r = p.add_run(label)
        r.bold = True
        cells[1].paragraphs[0].add_run(value)
    for row in table.rows:
        row.cells[0].width = Pt(150)
        row.cells[1].width = Pt(330)

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.add_run("Models implemented").bold = True
    for name, detail in MODELS:
        item = doc.add_paragraph(style="List Bullet")
        item.add_run(f"{name}: ").bold = True
        item.add_run(detail)

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.add_run("Contents of the repository").bold = True
    for line in [
        "Implementation as Python modules (src/) and a Colab-ready Jupyter "
        "notebook (notebooks/).",
        "README.md with the problem statement, dataset description and link, "
        "model descriptions, execution steps and dependencies, results, model "
        "comparison and evaluation, sampling-temperature analysis, and the "
        "conclusion identifying the best-performing model.",
        "Evaluation results in results/ - BLEU and chrF++ scores, training and "
        "inference times, parameter counts, failure-mode profiling, plots and "
        "the generated translations for the full test set.",
        "docs/responsible_ai.md - the responsible and ethical AI analysis "
        "covering bias, hallucination, cultural misrepresentation and sensitive "
        "content, with proposed mitigations.",
    ]:
        doc.add_paragraph(line, style="List Bullet")

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    print(f"[doc] wrote {path.relative_to(ROOT)}")
    return path


# --------------------------------------------------------------------------- #
# PDF conversion
# --------------------------------------------------------------------------- #

def _convert_with_word(docx_path: Path, pdf_path: Path) -> bool:
    """Microsoft Word via COM, driven through PowerShell so pywin32 is not needed."""
    if sys.platform != "win32":
        return False
    script = f"""
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {{
    $doc = $word.Documents.Open('{docx_path}', $false, $true)
    $doc.ExportAsFixedFormat('{pdf_path}', 17)
    $doc.Close($false)
}} finally {{
    $word.Quit()
}}
"""
    try:
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                        "-Command", script],
                       check=True, capture_output=True, timeout=180)
        return pdf_path.exists()
    except Exception as exc:
        print(f"[doc] Word conversion unavailable ({type(exc).__name__})")
        return False


def _convert_with_soffice(docx_path: Path, pdf_path: Path) -> bool:
    exe = shutil.which("soffice") or shutil.which("libreoffice")
    for candidate in (r"C:\Program Files\LibreOffice\program\soffice.exe",
                      r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"):
        if exe is None and Path(candidate).exists():
            exe = candidate
    if exe is None:
        return False
    try:
        subprocess.run([exe, "--headless", "--convert-to", "pdf",
                        "--outdir", str(pdf_path.parent), str(docx_path)],
                       check=True, capture_output=True, timeout=300)
        return pdf_path.exists()
    except Exception as exc:
        print(f"[doc] LibreOffice conversion failed ({type(exc).__name__})")
        return False


def _convert_with_docx2pdf(docx_path: Path, pdf_path: Path) -> bool:
    try:
        from docx2pdf import convert
        convert(str(docx_path), str(pdf_path))
        return pdf_path.exists()
    except Exception:
        return False


def convert_to_pdf(docx_path: Path) -> Path | None:
    pdf_path = docx_path.with_suffix(".pdf")
    if pdf_path.exists():
        pdf_path.unlink()
    for name, fn in (("Microsoft Word", _convert_with_word),
                     ("LibreOffice", _convert_with_soffice),
                     ("docx2pdf", _convert_with_docx2pdf)):
        if fn(docx_path, pdf_path):
            print(f"[doc] converted to PDF with {name}: "
                  f"{pdf_path.relative_to(ROOT)}")
            return pdf_path
    print("[doc] no converter available - open the .docx and use "
          "File -> Save As -> PDF")
    return None


def main() -> int:
    docx_path = build_docx(DOCS / f"{DOC_STEM}.docx")
    convert_to_pdf(docx_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
