# Neural Brains AI 3.0

Neural Brains AI is a personal project I've been working on to explore neural networks, conversational AI, and voice interaction.

I started this project to understand how different AI components can work together in one application. It currently has a Python backend, a browser-based dashboard, a custom neural network implementation, local memory storage, and Gemini API integration.

This is still a work in progress. I'm continuing to improve the application and try out new ideas as development moves forward.

## What it currently includes

- A custom Python neural network for basic classification.
- A text encoder that supports Unicode text.
- Gemini API integration for conversational responses.
- Browser-based voice input and text-to-speech.
- SQLite storage for conversation and experience memory.
- Model weight saving and loading.
- A dashboard for interacting with the application.
- An optional Ollama integration.

Some of these components are experimental, and there is still plenty of room for improvement.

## Technologies used

- Python
- HTML, CSS, and JavaScript
- Google Gemini API
- SQLite
- Web Speech API
- Ollama (optional)

## Project structure

```text
neural-brains-ai/
│
├── neural_brains/    # Python application
├── web/              # Dashboard
├── tests/            # Automated tests
├── docs/             # Project documentation
├── data/             # Local application data
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

The `data/` folder is created locally when the application needs it. It is not part of the public repository.

## Running the project locally

First, clone the repository:

```bash
git clone https://github.com/venureddy1133-netizen/neural-brains-ai.git
cd neural-brains-ai
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

The current Python application uses standard-library modules, so there are no third-party Python dependencies to install at this stage.

Configure your Gemini API key in a local `.env` file:

```text
GEMINI_API_KEY=your_api_key_here
```

You can optionally configure the Gemini model using `GEMINI_MODEL`.

Start the application:

```bash
python3 -m neural_brains --voice
```

Then open:

http://127.0.0.1:8765

Keep the terminal running while using the dashboard. Press `Ctrl + C` to stop the application.

## Running the tests

I've included tests for the neural network, text encoder, agent, conversation engine, memory, and dashboard.

Run them with:

```bash
python3 -m unittest discover -s tests -v
```

The initial test suite completed successfully with 18 passing tests.

## Files not included in this repository

I've intentionally left out some local files because they are not needed in a public source-code repository.

- `.env` – contains my private Gemini API key and local configuration.
- `.venv/` – my local Python environment.
- `data/` – local memory databases and saved model data.
- `__pycache__/` and `*.pyc` – generated Python cache files.
- Local backups, experiment files, and system-generated files.

These exclusions are managed through `.gitignore`.

The actual API key stays on my local machine. Anyone running this project will need to create their own `.env` file and provide their own API key.

## What's next?

I don't consider this a finished project. There are several things I'd like to work on as I continue developing it:

- Improve multilingual text handling and conversational responses.
- Make voice interaction more consistent across languages.
- Improve how conversation context and memory are managed.
- Experiment with neural network architecture and training.
- Continue working on the dashboard and its usability.
- Explore other AI model integrations.
- Add more tests and improve reliability.

I'll keep updating this repository as I make changes, add features, and test new ideas. The implementation may change along the way.

## Project status

**Version:** 3.0F – Neural

**Status:** Active development

**GitHub:** https://github.com/venureddy1133-netizen

---

This project is an ongoing learning and development effort. It is an experimental AI application, not a claim to have built artificial general intelligence or a human brain.
