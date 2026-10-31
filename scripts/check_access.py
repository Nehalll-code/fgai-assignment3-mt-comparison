"""Pre-flight: can this machine actually fetch the checkpoints the run needs?

By default the project loads ungated mirrors of Llama-3.2 and Gemma-2, so this
should pass with no Hugging Face token at all. It matters anyway, because the
pipeline does not crash on an unreachable checkpoint - it substitutes an open
stand-in and records that in ``results/run_info.json``. That is the right
behaviour for a run that must finish, and the wrong thing to discover two hours
in.

    python scripts/check_access.py

With FGAI_OFFICIAL_GATED=1 the official gated repos are checked instead; those
need an accepted licence and HF_TOKEN. The token value is never printed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src.decoder_only import can_access, hf_token  # noqa: E402


def targets() -> list[tuple[str, str, str]]:
    """(label, repo_id, licence page) for everything the run downloads."""
    return [
        ("mT5 (encoder-decoder)", C.P.mt5_id,
         f"https://huggingface.co/{C.P.mt5_id}"),
        ("Llama (decoder-only)", C.P.llama_id,
         f"https://huggingface.co/{C.P.llama_id}"),
        ("Gemma (decoder-only)", C.P.gemma_id,
         f"https://huggingface.co/{C.P.gemma_id}"),
    ]


def _diagnose(name: str, repo: str, has_token: bool) -> str:
    url = f"https://huggingface.co/{repo}"
    if name == "GatedRepoError":
        if not has_token:
            return (f"gated, and no token is set. Either accept the licence at "
                    f"{url} and set HF_TOKEN, or unset FGAI_OFFICIAL_GATED to "
                    "use the ungated mirror (no token needed).")
        return (f"gated, and the licence is not accepted (or is still pending) "
                f"on the account this token belongs to. Open {url} while signed "
                "in to that account and submit the access form.")
    if name in ("RepositoryNotFoundError", "HfHubHTTPError", "HTTPError"):
        if not has_token:
            return f"not reachable without a token. See {url}."
        return (f"the token cannot see {repo}. Either the licence at {url} is "
                "not accepted on the same account, or the token is "
                "fine-grained and is missing 'Read access to contents of all "
                "public gated repos you can access'.")
    return f"{name} - check the network, then {url}."


def main() -> int:
    token = hf_token()
    mode = "official gated repos" if C.USE_OFFICIAL_GATED else "ungated mirrors"
    print(f"checkpoint source: {mode}")
    print(f"HF token: {'present' if token else 'not set'}"
          + ("" if C.USE_OFFICIAL_GATED or token else " (not needed for mirrors)"))
    print()

    blocked = []
    for label, repo, _ in targets():
        ok, reason = can_access(repo, token)
        if ok:
            print(f"  OK       {label:24} {repo}")
        else:
            blocked.append(repo)
            print(f"  BLOCKED  {label:24} {repo}")
            print(f"           {_diagnose(reason, repo, bool(token))}")

    print()
    if not blocked:
        print("Everything the run needs is reachable.")
        if not C.USE_OFFICIAL_GATED:
            print("Note: Llama and Gemma are loaded from ungated mirrors of the "
                  "same weights.")
            print("      Which repo was used is recorded in "
                  "results/run_info.json and in the report.")
        return 0
    print("The pipeline will still finish, but anything blocked above is "
          "replaced by an open stand-in and the report will say so.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
