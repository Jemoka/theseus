# Web UI

The web app is a lightweight operational dashboard for jobs, logs, and checkpoints.

## Start the App

```bash
THESEUS_CLUSTER_ROOT=/path/to/root python -m theseus.web.app --port 8000
```

## What It Is Good For

- fast status checks without shelling into cluster hosts,
- browsing checkpoint trees and run metadata,
- reading logs from active and completed runs,
- quick project-level monitoring.

## Components

- service layer in `theseus/web/services/*`
- routes in `theseus/web/routes/*`
- templates in `theseus/web/templates/*`
- styling in `theseus/web/static/styles.css`

## Authentication

See `theseus/web/AUTH.md` for auth setup and password hash generation utility.

## Operational Notes

- Point the app at the same root used by your runs.
- Keep log/checkpoint directory conventions consistent across clusters for predictable display.
