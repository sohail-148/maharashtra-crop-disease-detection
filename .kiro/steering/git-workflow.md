---
inclusion: always
---

# Git Workflow Rules

## When the user says "push to GitHub"

Run the complete workflow automatically in this order:

1. `git status` and `git diff --stat` — understand what changed
2. Verify **none** of the following are staged or about to be staged:
   - Dataset image directories (`tomato_plantvillage/`, `grape_niphad/`, `grape_2024/`, `chilli_cold/`, `sugarcane_maharashtra/`, `sugarcane_large/`)
   - `venv/`
   - `*.keras`, `*.h5`, `*.ckpt`, `*.pb`, `*.tflite`, `*.onnx`, `saved_model/`, `models/`
   - `__pycache__/`, `*.pyc`
   - `.env`, secrets, credentials
   - Any file > 50 MB
3. `git add .` (or specific files if only a subset should be staged)
4. Write a concise, descriptive commit message based on the actual changes
5. `git commit -m "<message>"`
6. `git push origin main`
7. Report: commit hash, pushed branch, and final `git status`

## When NOT to commit or push

- Do **not** automatically commit or push unless the user explicitly says "push to GitHub" or equivalent
- Do not commit during normal coding/editing sessions

## Repository details

- Remote: `https://github.com/sohail-148/maharashtra-crop-disease-detection.git`
- Branch: `main`
- GitHub username: `sohail-148`
- Push requires a PAT if Git credential manager is not already cached

## Note on interactive authentication

If Git prompts for credentials interactively, provide the user with the exact command to run manually, substituting their PAT.
