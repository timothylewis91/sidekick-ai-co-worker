import gradio as gr
import re
from sidekick import Sidekick


# ============================================================================
# Initialization
# ============================================================================

async def setup():
    sk = Sidekick()
    await sk.setup()
    return sk


# ============================================================================
# Message Processing
# ============================================================================

async def process_message(sidekick, message, success_criteria, history):
    results = await sidekick.run_superstep(message, success_criteria, history)

    final_history = None
    metadata = None
    if isinstance(results, tuple) and len(results) == 2:
        final_history, metadata = results
    else:
        final_history = results

    if metadata:
        try:
            if "tools_used" in metadata:
                sidekick.tools_used = metadata["tools_used"] or []
            if "status" in metadata:
                sidekick.status = metadata["status"] or "Idle"
        except Exception:
            pass

    return final_history, sidekick


# ============================================================================
# Cleanup
# ============================================================================

def free_resources(sidekick):
    print("Cleaning up")
    try:
        if sidekick:
            sidekick.cleanup()
    except Exception as e:
        print(f"Exception during cleanup: {e}")


# ============================================================================
# UI Helpers
# ============================================================================

def _status_thinking():
    return gr.update(value="Thinking…")


def send_local_message(message_text, chat_history, full_history, show_eval_flag):
    # Append user message to authoritative full_history and compute displayed chat according to toggle
    if not message_text or not message_text.strip():
        return "", chat_history, full_history
    user_msg = {"role": "user", "content": f"**👤 User**\n\n{message_text.strip()}"}
    new_full = (full_history or []) + [user_msg]

    # Apply show_eval filter
    if not show_eval_flag:
        displayed = [m for m in new_full if not (isinstance(m.get("content", ""), str) and m.get("content", "").strip().startswith("🧠 **Evaluation"))]
    else:
        displayed = new_full

    return "", displayed, new_full


async def _process_and_update(sidekick, message, success_criteria, history, show_eval, full_history):
    input_message = message

    # If textbox is empty, use last user message already in chat
    if (not input_message or not str(input_message).strip()) and history:
        for item in reversed(history):
            if isinstance(item, dict) and item.get("role") == "user":
                input_message = item.get("content", "")
                input_message = re.sub(r"^\*\*👤 User\*\*\\n\\n", "", input_message)
                break

# Use full_history for running the agent and tracking evaluator feedback
        chat_history, sidekick_inst = await process_message(
        sidekick, input_message, full_history
    )

    # Ensure we have the authoritative full history (returned by run_superstep)
    new_full_history = chat_history

    tools_used = getattr(sidekick_inst, "tools_used", None)
    if not tools_used:
        tools_md = "No tools used yet."
    else:
        lines = []
        for t in tools_used:
            if isinstance(t, dict):
                ts = t.get("timestamp", "")
                name = t.get("name", "")
                disp = t.get("display_name", name)
                desc = t.get("description", "")
                src = t.get("source", "")
                detail_html = (
                    f"<details><summary>Details</summary><div class='tool-desc'>{desc}</div></details>"
                    if desc else ""
                )
                lines.append(f"- **{disp}** `({name})` — {src} @ {ts}\n{detail_html}")
            else:
                lines.append(f"- {t}")
        tools_md = "\n".join(lines)

    status = getattr(sidekick_inst, "status", "Idle")

    # Apply show_eval filter to produce a view for the Chatbot
    def _filter_eval_view(full_history_val, show_eval_flag):
        if not full_history_val:
            return full_history_val
        if show_eval_flag:
            return full_history_val
        filtered = []
        for item in full_history_val:
            content = item.get("content", "") if isinstance(item, dict) else ""
            if isinstance(content, str) and content.strip().startswith("🧠 **Evaluation"):
                continue
            filtered.append(item)
        return filtered

    # Use the authoritative new_full_history if present
    authoritative = new_full_history if 'new_full_history' in locals() else chat_history
    display_history = _filter_eval_view(authoritative, show_eval)

    return display_history, sidekick_inst, status, tools_md, authoritative


async def _reset_and_clear():
    new_sk = Sidekick()
    await new_sk.setup()
    return "", "", None, new_sk, "Idle", "No tools used yet."


# ============================================================================
# CSS (FORCE SIDE-BY-SIDE EVEN ON SMALL VIEWPORTS)
# ============================================================================

CUSTOM_CSS = """
/* Background */
html, body {
  background: radial-gradient(1200px 700px at 20% 0%,
    rgba(16,185,129,0.14) 0%,
    rgba(2,6,23,0.0) 55%),
    linear-gradient(180deg, #050b16 0%, #070a12 100%) !important;
}

/* Give the app room + allow horizontal scroll instead of stacking */
body { overflow-x: auto !important; }

/* Container */
.gradio-container {
  max-width: 1400px !important;
  margin: 0 auto !important;
  padding: 18px !important;

  /* this is the key: if viewport is narrow, we scroll, not stack */
  min-width: 1120px !important;
}

/* Panels */
.panel-card {
  border-radius: 16px !important;
  padding: 18px !important;
  background: linear-gradient(180deg,
    rgba(15,23,42,0.95) 0%,
    rgba(2,6,23,0.88) 100%) !important;
  border: 1px solid rgba(148,163,184,0.12) !important;
  box-shadow: 0 18px 40px rgba(0,0,0,0.45),
              inset 0 1px 0 rgba(255,255,255,0.04) !important;
  color: #e5e7eb !important;
}

/* Inputs */
.gr-textbox textarea, .gr-textbox input, textarea, input {
  border-radius: 12px !important;
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(148,163,184,0.14) !important;
  color: #e5e7eb !important;
}

/* Buttons */
.gr-button { border-radius: 12px !important; }

/* ============ THE IMPORTANT PART ============ */
/* Use high-specificity selectors AND override Gradio's mobile breakpoint rules */
.gradio-container .layout-row {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  align-items: stretch !important;
  gap: 16px !important;
}

/* If Gradio applies mobile styles, we re-override them here too */
@media (max-width: 1200px) {
  .gradio-container .layout-row {
    flex-direction: row !important;
    flex-wrap: nowrap !important;
  }
}

/* Left fixed width, right grows */
.gradio-container .left-panel {
  flex: 0 0 420px !important;
  width: 420px !important;
  max-width: 420px !important;
}

.gradio-container .right-panel {
  flex: 1 1 auto !important;
  min-width: 660px !important;
}

/* Chat sizing */
#chatbot {
  height: 640px !important;
  min-height: 640px !important;
  border-radius: 14px !important;
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(148,163,184,0.12) !important;
}

/* Row spacing */
.controls-row {
  gap: 10px;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
}

.tool-desc {
  color: #cbd5e1;
  font-size: 0.95rem;
  padding-left: 8px;
}
"""


# ============================================================================
# Gradio UI
# ============================================================================

theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="slate",
    neutral_hue="gray",
    radius_size="lg",
    text_size="md",
)

with gr.Blocks(title="Sidekick", theme=theme, css=CUSTOM_CSS) as ui:
    gr.Markdown("## Sidekick — Personal Co-Worker")

    sidekick = gr.State(delete_callback=free_resources)
    full_history = gr.State()

    # IMPORTANT: elem_classes is more reliable than elem_id for CSS overrides
    with gr.Row(elem_classes=["layout-row"]):
        with gr.Column(elem_classes=["panel-card", "left-panel"]):
            gr.Markdown("### Task")
            message = gr.Textbox(label="Task", placeholder="Describe the task...", lines=3)

            gr.Markdown("### Success Criteria")
            success_criteria = gr.Textbox(
                label="Success Criteria (short)",
                placeholder="List success criteria (one per line)...",
                lines=2,
            )

            status_box = gr.Textbox(label="Status", value="Idle", interactive=False, lines=1)

            with gr.Row(elem_classes=["controls-row"]):
                send_btn = gr.Button("Send", variant="primary")
                go_button = gr.Button("Go!", variant="primary")
                reset_button = gr.Button("Reset", variant="stop")

            with gr.Accordion("Sidekick used tools (toggle)", open=False):
                tools_text = gr.Markdown("No tools used yet.")

            # Toggle to show/hide evaluator feedback
            show_eval = gr.Checkbox(label="Show evaluator feedback", value=True)
        with gr.Column(elem_classes=["panel-card", "right-panel"]):
            gr.Markdown("### Conversation")
            chatbot = gr.Chatbot(elem_id="chatbot", type="messages")

    # Events
    ui.load(setup, [], [sidekick])

    message.submit(send_local_message, [message, chatbot, full_history, show_eval], [message, chatbot, full_history])
    send_btn.click(send_local_message, [message, chatbot, full_history, show_eval], [message, chatbot, full_history])

    go_button.click(_status_thinking, [], [status_box])
    success_criteria.submit(_status_thinking, [], [status_box])

    go_button.click(
        _process_and_update,
        [sidekick, message, success_criteria, chatbot, show_eval, full_history],
        [chatbot, sidekick, status_box, tools_text, full_history],
    )
    success_criteria.submit(
        _process_and_update,
        [sidekick, message, success_criteria, chatbot, show_eval, full_history],
        [chatbot, sidekick, status_box, tools_text, full_history],
    )

    # Show/hide evaluator feedback when the toggle changes
    def _toggle_eval_view(show_eval_flag, fh):
        if not fh:
            return None
        if show_eval_flag:
            return fh
        filtered = []
        for item in fh:
            content = item.get("content", "") if isinstance(item, dict) else ""
            if isinstance(content, str) and content.strip().startswith("🧠 **Evaluation"):
                continue
            filtered.append(item)
        return filtered

    show_eval.change(_toggle_eval_view, [show_eval, full_history], [chatbot])

    reset_button.click(
        _reset_and_clear,
        [],
        [message, success_criteria, chatbot, sidekick, status_box, tools_text, full_history],
    )

ui.launch(inbrowser=True)
