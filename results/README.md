# results/

This directory is **generated**. It is empty in a fresh clone and is filled by

```bash
python -m src.run_all          # or the notebook on a GPU runtime
python scripts/update_readme.py
```

What lands here after a full run:

| File | Contents |
|---|---|
| `metrics.csv` / `.md` / `.json` | BLEU, chrF++, parameter counts, training time, inference time and ms/sentence for every system |
| `temperature_sweep.csv` / `.md` | the sampling-temperature analysis |
| `error_analysis.csv` / `.md` | failure-mode profiling (seven flags) per system |
| `error_examples.csv` | concrete instances of each flag, plus the worst sentences |
| `responsible_ai_probes.csv` / `.md` | probe-set outputs used in the responsible-AI discussion |
| `sample_translations.md` | the first 15 test sentences, all systems side by side |
| `test_set_en_hi.jsonl` | the exact 500 held-out pairs the scores are computed on |
| `run_info.json` | device, library versions, split sizes, sacrebleu signatures |
| `translations/*.csv` | every hypothesis with its sentence-level chrF++ |
| `plots/*.png` | the figures embedded in the README |

The root `README.md` injects its numbers from these files, so the report always
describes the run that actually happened.
