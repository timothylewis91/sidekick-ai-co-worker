# Sidekick — Personal Co-Worker 

💡 **Short summary**

Sidekick is a self-evaluating AI assistant that uses a worker→tools→evaluator loop to complete tasks, iterate when feedback requires it, and ask for clarification as needed. This README documents how to run, test, deploy, and scale the app (recommended deployment: Hugging Face Spaces + GitHub CI).

---

## 🎯 Problem Statement & Target User

- **Problem:** People need a reliable *multi-step* agent that can use tools (browser, Python, search, files) and evaluate its own answers against user-specified success criteria.
- **Target user:** Researchers and small teams who want an extensible local or hosted assistant for complex, tool-enabled tasks.

---

## ✅ Features

- Worker LLM with tool bindings (Playwright, search, Python REPL, files, push notifications)
- Evaluator LLM that returns structured feedback and decides whether to continue or finish
- Gradio-based UI for task input, success criteria, and conversation
- Tool usage metadata and UI transparency panel
- Configurable, modular codebase intended for easy extension and productionization

---

## Quickstart (Local)

Requirements:
- Python 3.11+ (3.12 tested in this workspace)
- Virtualenv or venv

Steps:

```bash
# create and activate a venv
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# export required env vars (example)
copy .env.example .env  # edit tokens/keys

# run the app
python app.py
# or: uv run app.py for uvicorn-like runner (if available)
```

Open http://127.0.0.1:7860 in your browser.

> Tip: Use `requirements.txt` or `pyproject.toml` for dependency management.

---

## Configuration & ENV

Important environment variables (example):
- SERPER_API_KEY — Google Serper (search)
- PUSHOVER_TOKEN, PUSHOVER_USER — push notifications
- Other LLM keys (if applicable) — set through `.env` or platform secrets

Keep secrets out of Git!

---

## Deployment plan (GitHub Actions + optional container registry)

Goal: Run tests on every push and provide a simple way to publish a runnable artifact. By default the repository's CI runs tests (`pytest`) on push and manual dispatch. If you'd like deployment or packaging, use one of the options below:

- GitHub Container Registry (recommended): Build and publish a Docker image to GHCR and pull/run it anywhere. This is a portable, GitHub-native approach and requires no extra provider accounts.
- Custom host: Build a Docker image and deploy it to your cloud provider (Render, DigitalOcean, AWS, etc.) using provider-specific deployment steps or provider-specific GitHub Actions.

Notes:
- The repository includes a `Dockerfile` (if present) or you can add one to containerize the app.
- The CI currently only runs tests; if you'd like, I can add a `build-and-push` job that publishes to GHCR on push to `main` or on manual trigger.
- The previous Hugging Face Spaces deploy instructions have been removed from this README.

Additions:
- The workflow supports manual runs via `workflow_dispatch`.
- If you want me to add GHCR publishing, tell me and I’ll add a small job that builds `ghcr.io/${{ github.repository_owner }}/${{ github.repository }}:sidekick-<sha>` on main or on demand.


Notes:
- Spaces can run Gradio apps directly. For heavier loads, consider a Docker-based deployment on cloud providers.
- Choose a hardware configuration in Spaces if you require GPU acceleration.

---

## Metrics to Track (Success Criteria)

Metrics help evaluate the product and the agent:
- **Task success rate:** % of runs where evaluator returns `success_criteria_met=True` (primary KPI)
- **Latency:** time to first assistant response and time to final result
- **Tool usage:** frequency of each tool (search, python, playwright) and average time per tool
- **Error rate:** exceptions, failed tool calls, browser crashes
- **Cost:** token usage / API call cost per run
- **User satisfaction:** explicit feedback or thumbs up/down per finished task

Record these in logs and emit structured telemetry to a monitoring backend (Prometheus, Datadog, or a simple CSV during development).

---

## Architecture & Modularization Suggestions

Proposed layout (refactor if not yet present):

```
4_langgraph/
  app.py                # Gradio UI (keeps only UI wiring)
  sidekick.py           # Core Sidekick class (graph, run_superstep)
  sidekick_tools.py     # Tool builders + metadata helpers
  src/                  # Optional: core library code (Sidekick, utils)
  tests/                # Unit and integration tests
  README.md
  requirements.txt
  .github/workflows/    # CI & CD workflows
```

Guidelines:
- Keep UI code (Gradio) thin — delegate logic to Sidekick and tool modules.
- `sidekick_tools.py` should expose a normalized metadata map for each tool. Use standardized return values for tool outputs.
- Use dependency inversion (small interfaces) so you can swap LLM implementations, tool backends, or the checkpointer.
- Add unit tests for tool wrappers and integration tests for full runs with mock LLMs.

---

## Scalability & Production Readiness

Short-term (safe):
- Containerize (`Dockerfile`) and pin dependency versions
- Add a basic health endpoint and graceful shutdown handlers
- Track metrics (Prometheus) and export logs (structured JSON)

Long-term (more effort):
- Make agent runs asynchronous and move long-running runs to a worker queue (Redis + RQ / Celery)
- Use a shared state store (Redis) for conversation threads in multi-instance deployments
- Rate limit and queue GPU-bound models; separate tool executors into microservices
- Add role-based authentication and per-user namespaces

---

## Testing & QA

- Unit tests for `sidekick_tools` and `sidekick` logic
- Integration tests that mock or stub external APIs (Serper, Playwright)
- End-to-end tests driving the Gradio UI using Playwright or `gradio_client`

---


