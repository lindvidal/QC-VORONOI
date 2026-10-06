---
trigger: always_on
---

# Auto Commit & Push Policy

After completing modifications requested by the user:
1. Determine the next ordinal commit version based on `git rev-list --count HEAD` (4 digits, e.g. `v0004`).
2. Format commit messages strictly starting with `v0000 | ` in an ordinal manner:
   Format: `v{####} | <type>(<scope>): <resumo>`
   Example: `v0004 | feat(pipeline): implement spatial delaunay point-out mesh`
   Followed by detailed, verbose description body adhering to conventional commits.
3. Always generate and update the telemetry logs directly to docs (`log_github_lastrun.csv` and `log_github_lastrun.docx`) via `python src/services/github_logger.py` without generating temporary scripts.
4. Execute `git add`, `git commit` and automatically `git push` to origin.
