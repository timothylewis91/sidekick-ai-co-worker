# Sidekick (minimal)

This branch contains a minimal version of the Sidekick app for demonstration purposes.

Included:
- `4_langgraph/app.py`
- `4_langgraph/sidekick.py`
- `4_langgraph/sidekick_tools.py`
- `4_langgraph/README.md`
- `4_langgraph/sandbox/` (sample content)

How to run locally:

1. Create and activate a virtualenv:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.minimal.txt
python 4_langgraph/app.py
```

Or using Docker:

```bash
docker build -t sidekick:minimal -f Dockerfile.minimal .
docker run --rm -p 7860:7860 sidekick:minimal
```

Note: The `requirements.minimal.txt` contains a minimal set of dependencies and may need to be adjusted for your environment.
