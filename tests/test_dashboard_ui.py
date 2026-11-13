import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "web" / "index.html"

class DashboardUITests(unittest.TestCase):
    def test_dashboard_has_all_sections(self):
        content = HTML.read_text(encoding="utf-8")
        for marker in (
            'data-page="overview"', 'data-page="voice"', 'data-page="memory"',
            'data-page="network"', 'data-page="settings"', 'id="distribution"',
            'id="memoryRows"', 'id="fullNetwork"', 'id="refresh"',
            'id="geminiStatus"', 'GEMINI_API_KEY', 'Matching local memories are not sent by default.',
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, content)

    def test_dashboard_uses_local_api_and_safe_text_rendering(self):
        content = HTML.read_text(encoding="utf-8")
        self.assertIn("fetch('/api/health')", content)
        self.assertIn("fetch('/api/stats')", content)
        self.assertIn("fetch('/api/memory')", content)
        self.assertIn("node.textContent=text", content)
        self.assertIn("window.setInterval(refreshData,15000)", content)

if __name__ == "__main__":
    unittest.main()
