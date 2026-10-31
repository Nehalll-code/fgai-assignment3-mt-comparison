"""Pre-flight check: can this token actually fetch the two gated checkpoints?

Both Llama-3.2 and Gemma-2 are gated on Hugging Face. If the licence has not
been accepted, or the token lacks gated-repo permission, the pipeline does not
crash - it falls back to an open stand-in and records that in
``results/run_info.json``. That is the right behaviour for a run that must
finish, but it is the wrong thing to discover two hours in, so this check runs
first and says exactly what is wrong.

    python scripts/check_access.py

Reads the token from HF_TOKEN (or HUGGING_FACE_HUB_TOKEN). The token itself is
never printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GATED = [
    ("meta-llama/Llama-3.2-1B-Instruct",
     "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct"),
    ("google/gemma-2-2b-it",
     "https://huggingface.co/google/gemma-2-2b-it"),
]


def check(token: str | None) -> list[tuple[str, bool, str]]:
    # Deliberately the same helper the pipeline uses to decide whether to fall
    # back, so this check cannot disagree with what the run will actually do.
    from src.decoder_only import can_access

    results = []
    for repo, url in GATED:
        ok, reason = can_access(repo, token)
        results.append((repo, ok, "" if ok else _diagnose(reason, url, bool(token))))
    return results


def _diagnose(name: str, url: str, has_token: bool) -> str:
    if name == "GatedRepoError":
        if not has_token:
            return (f"gated, and no token is set. Accept the licence at {url} "
                    "while signed in, then set HF_TOKEN.")
        return (f"licence not accepted (or still pending) on the account this "
                f"token belongs to. Open {url} while signed in to that account "
                "and submit the access form.")
    if name in ("RepositoryNotFoundError", "HfHubHTTPError", "HTTPError"):
        if not has_token:
            return "no token supplied - set HF_TOKEN."
        return (f"the token cannot see this repo. Either the licence at {url} "
                "is not accepted on the same account, or the token is "
                "fine-grained and is missing 'Read access to contents of all "
                "public gated repos you can access'.")
    return f"{name} - check your network, then the licence at {url}."


def main() -> int:
    import os

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"token: {'present' if token else 'NOT SET'}")

    ok_all = True
    for repo, ok, why in check(token):
        if ok:
            print(f"  OK       {repo}")
        else:
            ok_all = False
            print(f"  BLOCKED  {repo}")
            print(f"           {why}")

    print()
    if ok_all:
        print("Both gated checkpoints are reachable. The run will use the real "
              "Llama and Gemma.")
        return 0
    print("The pipeline will still run, but it will substitute open stand-in "
          "checkpoints for whatever is blocked above, and the report will say "
          "so. Fix the access first if the assignment needs the real models.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
