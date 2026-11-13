import tempfile,unittest
from neural_brains.agent import BrainAgent, BOOTSTRAP
class AgentTests(unittest.TestCase):
 def test_bootstrap_examples_classify(self):
  with tempfile.TemporaryDirectory() as d:
   a=BrainAgent(d)
   for text,label in BOOTSTRAP:
    self.assertEqual(a.think(text)['label'],label, f"failed starter example: {text}")
 def test_think_and_persistence(self):
  with tempfile.TemporaryDirectory() as d:
   a=BrainAgent(d); r=a.think("hello there"); self.assertEqual(r['label'],"greeting"); a.teach("please create a data report","task",35)
   expected=a.network.predict(a.encoder.encode("please create a data report")); b=BrainAgent(d); actual=b.network.predict(b.encoder.encode("please create a data report")); self.assertEqual(expected,actual); self.assertGreaterEqual(b.memory.count(),2)
 def test_bad_label(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError): BrainAgent(d).teach("hello","unknown")
if __name__=="__main__": unittest.main()
