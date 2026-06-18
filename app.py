import os
import tempfile
import modal
import gradio as gr
import soundfile as sf
from dotenv import load_dotenv
from modal import Cls

load_dotenv()

# --------------------------------------------------
# Clients
# --------------------------------------------------

CodingCompanion = Cls.from_name("coding-companion", "CodingCompanion")
modal_model = CodingCompanion()

# --------------------------------------------------
# Personas
# --------------------------------------------------

PERSONAS = {
    "Senior Engineer": """
You are a blunt, experienced senior software engineer doing a code review or technical discussion.
You never just answer — you always question the approach first.
Ask why they made a choice before suggesting alternatives.
Be direct, sometimes terse. Push back on suboptimal decisions.
Reference time/space complexity naturally.
One or two pointed questions per response. Never hand-hold.
""",
    "Rubber Duck": """
You are a rubber duck debugging assistant.
You never give answers or suggestions.
You only ask ONE clarifying question per response to help the person think through their own problem.
Be warm and patient. Your job is to make them arrive at the answer themselves.
Never reveal the solution. Never say "good job".
Just keep asking the next logical question.
""",
    "FAANG Interviewer": """
You are a FAANG technical interviewer conducting a coding interview. Stay strictly in character.
After they explain their approach, probe edge cases, ask about complexity, push for optimization.
Apply light pressure — this is an interview.
Phrases like "interesting, but what happens if..." or "can we do better than O(n²)?"
Never be cruel, but never let a weak answer slide.
"""
}

# --------------------------------------------------
# Audio Transcription
# --------------------------------------------------

def transcribe_audio(audio_data):
    if audio_data is None:
        return ""

    sample_rate, audio_array = audio_data

    # Convert to wav bytes in memory
    import io
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio_array, sample_rate, format="WAV")
    audio_bytes = buf.getvalue()

    return modal_model.transcribe.remote(audio_bytes)

# --------------------------------------------------
# Model Call
# --------------------------------------------------

def get_response(persona_name, chat_history, user_message):
    system_prompt = PERSONAS[persona_name]

    messages = [{"role": "system", "content": system_prompt}]

    for msg in chat_history:
        if msg["role"] in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    return modal_model.generate.remote(messages)

# --------------------------------------------------
# Chat Handler
# --------------------------------------------------

def chat(user_text, audio_input, persona_choice, history):
    if user_text and user_text.strip():
        message = user_text.strip()
    elif audio_input is not None:
        message = transcribe_audio(audio_input)
        if not message:
            return history, "", None, "⚠️ Couldn't transcribe audio."
    else:
        return history, "", None, "⚠️ Please type or speak something."

    response = get_response(persona_choice, history, message)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    return history, "", None, ""

# --------------------------------------------------
# UI
# --------------------------------------------------

theme = gr.themes.Base(
    primary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"]
)

css = """
.gradio-container { max-width: 860px !important; margin: auto; }
#title { text-align: center; padding: 1rem 0 0.25rem 0; }
#subtitle { text-align: center; color: #94a3b8; margin-bottom: 1.5rem; font-size: 0.9rem; }
#error-box { color: #f87171; font-size: 0.85rem; min-height: 1.2rem; }
"""

with gr.Blocks() as demo:

    gr.Markdown("# 🧑‍💻 Interview Companion", elem_id="title")
    gr.Markdown(
        "Your solo FAANG prep buddy — speak or type, pick a persona, get real feedback.",
        elem_id="subtitle"
    )

    persona_selector = gr.Radio(
        choices=list(PERSONAS.keys()),
        value="😬 FAANG Interviewer",
        label="Choose your companion"
    )

    chatbot = gr.Chatbot(height=500)

    audio_input = gr.Audio(
        sources=["microphone"],
        type="numpy",
        label="🎙️ Speak your approach"
    )

    with gr.Row():
        text_input = gr.Textbox(
            placeholder="...or type your code / approach here",
            lines=3,
            scale=5
        )
        send_btn = gr.Button("Send →", variant="primary", scale=1)

    error_display = gr.Markdown("", elem_id="error-box")
    history_state = gr.State([])

    send_btn.click(
        fn=chat,
        inputs=[text_input, audio_input, persona_selector, history_state],
        outputs=[chatbot, text_input, audio_input, error_display]
    ).then(lambda h: h, chatbot, history_state)

    text_input.submit(
        fn=chat,
        inputs=[text_input, audio_input, persona_selector, history_state],
        outputs=[chatbot, text_input, audio_input, error_display]
    ).then(lambda h: h, chatbot, history_state)

    gr.Markdown(
        "💡 **Tips:** Describe your approach out loud like you would in a real interview. "
        "Switch personas mid-session to get a different angle."
    )

demo.launch(theme=theme, css=css)