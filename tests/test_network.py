import tempfile,unittest
from pathlib import Path
from neural_brains.network import NeuralNetwork
class NetworkTests(unittest.TestCase):
 def test_softmax(self):
  p=NeuralNetwork().predict([0.0]*64); self.assertAlmostEqual(sum(p),1.0,places=7); self.assertTrue(all(0<=x<=1 for x in p))
 def test_train_save_load(self):
  m=NeuralNetwork(); x=[.1]*64; before=m.predict(x)
  for _ in range(30): m.train_one(x,1)
  self.assertGreater(m.predict(x)[1],before[1])
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/"model.json"; m.save(path); n=NeuralNetwork.load(path); self.assertEqual(m.predict(x),n.predict(x)); self.assertEqual(m.training_updates,n.training_updates)
 def test_bad_vector(self):
  with self.assertRaises(ValueError): NeuralNetwork().predict([1.0,2.0])
if __name__=="__main__": unittest.main()
