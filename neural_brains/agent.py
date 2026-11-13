"""Persistent brain agent: classify, learn from corrections, and recall experiences."""
from pathlib import Path
import random
from .encoder import TextEncoder
from .memory import MemoryStore
from .network import NeuralNetwork
LABELS=("greeting","question","task")
BOOTSTRAP=[("hello there","greeting"),("hi how are you","greeting"),("good morning","greeting"),("hey nice to meet you","greeting"),("good evening friend","greeting"),("hello","greeting"),("what is machine learning","question"),("why is the sky blue","question"),("how does a neural network work","question"),("when is the meeting","question"),("where can I find the report","question"),("explain this concept","question"),("build a python application","task"),("write a function for me","task"),("create a project dashboard","task"),("fix this program","task"),("generate a report","task"),("help me deploy this app","task")]
class BrainAgent:
    def __init__(self,data_dir="data"):
        self.data_dir=Path(data_dir); self.data_dir.mkdir(parents=True,exist_ok=True); self.model_path=self.data_dir/"brain_model.json"
        self.encoder=TextEncoder(64); self.memory=MemoryStore(self.data_dir/"brain_memory.sqlite3")
        if self.model_path.exists(): self.network=NeuralNetwork.load(self.model_path); self.model_loaded=True
        else:
            self.network=NeuralNetwork(64,24,len(LABELS)); self.model_loaded=False
            # Train across shuffled examples so later classes do not overwrite earlier ones.
            dataset=[(self.encoder.encode(text),LABELS.index(label)) for text,label in BOOTSTRAP]
            rng=random.Random(23)
            for _ in range(300):
                rng.shuffle(dataset)
                for vector,target in dataset:
                    self.network.train_one(vector,target,learning_rate=0.015)
            self.network.save(self.model_path)
    def _train(self,text,label,epochs=1,save=True):
        if label not in LABELS: raise ValueError(f"label must be one of: {', '.join(LABELS)}")
        vector=self.encoder.encode(text); losses=[self.network.train_one(vector,LABELS.index(label),learning_rate=0.03) for _ in range(epochs)]
        if save: self.network.save(self.model_path)
        return sum(losses)/len(losses)
    def think(self,text):
        if not isinstance(text,str) or not text.strip(): raise ValueError("Please enter non-empty text")
        ranked=sorted(zip(LABELS,self.network.predict(self.encoder.encode(text))),key=lambda x:x[1],reverse=True)
        self.memory.remember(text,ranked[0][0],source="prediction")
        return {"input":text,"label":ranked[0][0],"scores":dict(ranked),"memories":self.memory.recall(text)}
    def teach(self,text,label,epochs=25):
        if not isinstance(text,str) or not text.strip(): raise ValueError("Training text cannot be empty")
        loss=self._train(text,label,epochs,save=True); self.memory.remember(text,label,source="teaching"); return loss
    def stats(self): return {"experiences":self.memory.count(),"training_updates":self.network.training_updates,"model_file":str(self.model_path),"model_loaded":self.model_loaded}
