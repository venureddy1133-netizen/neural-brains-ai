# Phase 3 implementation notes

## What was added
- Browser microphone input using Web Speech API.
- Push-to-talk using the microphone control.
- Optional wake-word phrase matching for "Hey Neural Brains" while the page is open.
- Spoken assistant responses using browser speech synthesis.
- Local Python HTTP API bound to 127.0.0.1.
- Optional Ollama integration for local open-ended conversation.
- Built-in fallback when Ollama is not running.

## Known limitations
- Browser speech recognition availability and offline behavior depend on browser/OS. Chrome may use a remote service.
- Wake-word mode is phrase matching on browser-recognized speech, not a dedicated always-on low-power wake-word detector.
- The existing 64-24-3 neural classifier remains an intent classifier, not the conversational language model.
- Without Ollama, open-ended responses are intentionally limited.
