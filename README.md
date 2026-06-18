# Vocalize — Interview Companion

An entirely local, voice-first AI mock interviewer to practice technical communication

## Quick Setup (Windows PowerShell)
Create and activate a virtual environment, then install dependencies:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a .env in the project root for any environment variables you need:
```text
# .env (example)
# Add any keys your app or Modal needs, e.g. API keys
# EXAMPLE_KEY=your_value_here
```

## Run (local development)
Start the Gradio app:
```powershell
python app.py
```
Gradio will print a local URL (and a public share URL if enabled) where you can open the UI.

Notes:
- app.py expects a running Modal backend referenced via the `modal` client (the code uses `Cls.from_name` to obtain the remote class). For pure local development you can:
  - Mock `modal_model` with a lightweight local implementation that exposes `transcribe.remote()` and `generate.remote()` methods, or
  - Replace those calls temporarily with local stubs to test the UI and audio handling.

## Modal backend
- modal_backend.py defines a Modal `App` and the `CodingCompanion` class which loads a language model and tiny Whisper for transcription. Deploying this requires Modal credentials and following Modal's deployment docs.
- If you plan to deploy to Modal, ensure you have Modal configured locally and the GPU/image requirements met.

## Audio handling
- The Gradio `Audio` component uses `type="numpy"` and `soundfile` for conversions. Ensure `soundfile` is installed and functional on your platform.
- Audio transcription flow:
  - Gradio provides a `(sample_rate, numpy_array)` tuple.
  - app.py writes the numpy array to an in-memory WAV via `soundfile` and sends bytes to the backend for transcription.

## Testing
Run tests (if you use pytest):
```powershell
python -m pytest -q
```
