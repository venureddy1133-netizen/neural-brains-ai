import unittest
from neural_brains.encoder import TextEncoder
class EncoderTests(unittest.TestCase):
 def test_deterministic_normalized(self):
  e=TextEncoder(); v=e.encode("Neural brains learn"); self.assertEqual(v,e.encode("Neural brains learn")); self.assertAlmostEqual(sum(x*x for x in v),1.0,places=6)
 def test_empty(self): self.assertEqual(TextEncoder().encode(""),[0.0]*64)
if __name__=="__main__": unittest.main()
