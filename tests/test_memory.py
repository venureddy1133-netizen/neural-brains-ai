import tempfile,unittest
from neural_brains.memory import MemoryStore
class MemoryTests(unittest.TestCase):
 def test_remember_recall(self):
  with tempfile.TemporaryDirectory() as d:
   m=MemoryStore(d+"/memory.sqlite3"); m.remember("neural network learning","question"); self.assertEqual(m.count(),1); self.assertEqual(m.recall("neural network",1)[0]['label'],"question")
if __name__=="__main__": unittest.main()
