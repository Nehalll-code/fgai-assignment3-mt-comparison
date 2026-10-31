"""Fold a finished GPU run back into the repository, in one command.

The Colab notebook ends by producing ``fgai_results.zip``.  This script takes
that zip (or an already-extracted directory), installs it over ``results/``,
re-injects every number into ``README.md`` and ``docs/responsible_ai.md``, and
rebuilds the submission .docx and .pdf so they carry the best-performing model.

    python scripts/import_results.py ~/Downloads/fgai_results.zip

A zip is an archive from elsewhere, so every member path is checked before
anything is written: absolute paths, drive letters and ``..`` segments are
refused, and only paths inside ``results/`` plus the two report files are
accepted.  Nothing is extracted until the whole archive has passed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Everything the notebook's zip is allowed to contain.
ALLOWED_PREFIXES = ("results/",)
ALLOWED_FILES = ("README.md", "docs/responsible_ai.md")


class UnsafeArchive(Exception):
    """The archive tried to write somewhere it has no business writing."""


def _check_member(name: str) -> str | None:
    """Return the normalised path to extract, or None to skip it silently."""
    if name.endswith("/"):
        return None
    pure = PurePosixPath(name.replace("\\", "/"))
    if pure.is_absolute() or (len(pure.parts) > 0 and ":" in pure.parts[0]):
        raise UnsafeArchive(f"absolute path in archive: {name!r}")
    if ".." in pure.parts:
        raise UnsafeArchive(f"parent-directory escape in archive: {name!r}")

    text = str(pure)
    if text.startswith(ALLOWED_PREFIXES) or text in ALLOWED_FILES:
        return text
    return None


def _plan(names: list[str]) -> list[tuple[str, str]]:
    """Validate every member first; extract nothing until the whole set passes."""
    plan = []
    for name in names:
        target = _check_member(name)
        if target:
            plan.append((name, target))
    if not plan:
        raise UnsafeArchive(
            "archive contains nothing under results/ - is this the right file?")
    return plan


def import_zip(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        plan = _plan(zf.namelist())
        written = []
        for name, target in plan:
            dest = ROOT / target
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
            written.append(target)
    return written


def import_dir(src_dir: Path) -> list[str]:
    """Same contract as import_zip, for an already-extracted download."""
    base = src_dir / "results" if (src_dir / "results").is_dir() else src_dir
    if base.name != "results":
        raise UnsafeArchive(f"{src_dir} does not contain a results/ directory")
    written = []
    for path in sorted(base.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(base)
        dest = ROOT / "results" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        written.append(str(PurePosixPath("results") / PurePosixPath(rel.as_posix())))
    return written


def summarise(written: list[str]) -> bool:
    """Print what landed and whether it came from a real run. True if full."""
    print(f"[import] wrote {len(written)} file(s) into {ROOT.name}/")
    for group, label in (("results/plots/", "plots"),
                         ("results/translations/", "translation files")):
        n = sum(1 for w in written if w.startswith(group))
        if n:
            print(f"[import]   {n} {label}")

    info_path = ROOT / "results" / "run_info.json"
    if not info_path.exists():
        print("[import] WARNING: no results/run_info.json - cannot tell which "
              "profile produced these numbers")
        return False

    info = json.loads(info_path.read_text(encoding="utf-8"))
    device = info.get("gpu") or info.get("device", "?")
    print(f"[import] run: profile={info.get('profile')} device={device} "
          f"finished={info.get('finished_utc')} "
          f"wall clock={info.get('wall_clock_s', 0) / 60:.0f} min")

    fallbacks = [name for name, ck in (info.get("checkpoints") or {}).items()
                 if ck.get("is_fallback")]
    if fallbacks:
        print("[import] NOTE: these systems used an open fallback checkpoint "
              "rather than the gated one: " + ", ".join(fallbacks))

    if info.get("profile") != "full":
        print(f"[import] WARNING: this is the '{info.get('profile')}' profile. "
              "Those scores are not meaningful - the README will say so.")
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path,
                    help="fgai_results.zip, or a directory containing results/")
    ap.add_argument("--no-doc", action="store_true",
                    help="skip rebuilding the submission .docx/.pdf")
    args = ap.parse_args(argv)

    source = args.source.expanduser()
    if not source.exists():
        print(f"[import] no such file: {source}")
        return 1

    try:
        written = (import_zip(source) if zipfile.is_zipfile(source)
                   else import_dir(source))
    except UnsafeArchive as exc:
        print(f"[import] refused: {exc}")
        return 1

    is_full = summarise(written)

    from scripts import update_readme
    for target in (ROOT / "README.md", ROOT / "docs" / "responsible_ai.md"):
        if update_readme.fill(target):
            print(f"[import] refreshed {target.relative_to(ROOT)}")

    if not args.no_doc:
        from scripts import make_submission_doc
        make_submission_doc.main()

    print("\n[import] next:")
    print("  git add -A && git commit -m 'results: full GPU run' && git push")
    if not is_full:
        print("  (but re-run the full profile on a GPU first - these numbers "
              "are placeholders)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
