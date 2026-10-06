---
trigger: always_on
---

# Auto Commit & Push Policy

After completing modifications requested by the user:
1. Always generate and update the telemetry logs directly to docs (`log_github_lastrun.csv` and `log_github_lastrun.docx`) via `python src/services/github_logger.py` without generating temporary scripts.
2. Execute `git add`, `git commit` with verbose, descriptive commit messages adhering to conventional commits.
3. Automatically execute `git push` to origin.
