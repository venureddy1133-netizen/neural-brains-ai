"""Command-line entry point for Neural Brains AI 3.0."""
import argparse
from .agent import BrainAgent, LABELS
from .dashboard import build_dashboard

def main():
    parser = argparse.ArgumentParser(description="Neural Brains AI 3.0")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--voice", action="store_true",
                        help="Start the local browser voice assistant")
    args = parser.parse_args()

    if args.voice:
        from .server import run
        run(data_dir=args.data_dir)
        return

    agent = BrainAgent(args.data_dir)
    if args.dashboard:
        print(f"Dashboard generated: {build_dashboard(agent).resolve()}")
        return
    if args.demo:
        for text in ("hello there", "Why do neural networks learn?", "Create a Python project"):
            result = agent.think(text)
            print(f"\\nInput: {text}\\nPrediction: {result['label']}")
            print("Scores: " + ", ".join(
                f"{key}={value:.1%}" for key, value in result["scores"].items()
            ))
        print(f"\\nDashboard: {build_dashboard(agent).resolve()}\\nStats: {agent.stats()}")
        return

    print("\\nNEURAL BRAINS AI 3.0")
    print("Commands: think, teach, memory, dashboard, stats, quit")
    while True:
        command = input("\\nbrain> ").strip().lower()
        if command in ("quit", "exit", "q"):
            break
        try:
            if command == "think":
                result = agent.think(input("Text: ").strip())
                print(f"Prediction: {result['label']}")
                for label, score in result["scores"].items():
                    print(f"  {label:10s} {score:.1%}")
            elif command == "teach":
                text = input("Example text: ").strip()
                label = input(f"Correct label {LABELS}: ").strip().lower()
                loss = agent.teach(text, label)
                print(f"Learned and saved. Average training loss: {loss:.4f}")
            elif command == "memory":
                for item in agent.memory.recent(15):
                    print(f"[{item['label']}/{item['source']}] {item['text']}")
            elif command == "dashboard":
                print(f"Generated: {build_dashboard(agent).resolve()}")
            elif command == "stats":
                print(agent.stats())
            elif command:
                print("Unknown command.")
        except (ValueError, OSError) as exc:
            print(f"Error: {exc}")

if __name__ == "__main__":
    main()
