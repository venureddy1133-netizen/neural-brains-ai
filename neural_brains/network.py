"""A compact single-hidden-layer neural network with softmax output."""
import json, math, random
from pathlib import Path
MODEL_VERSION = 1
def _softmax(values):
    peak=max(values); exps=[math.exp(max(-60.0,min(60.0,v-peak))) for v in values]; total=sum(exps)
    return [v/total for v in exps]
class NeuralNetwork:
    def __init__(self,input_size=64,hidden_size=24,output_size=3,seed=17):
        self.input_size,self.hidden_size,self.output_size=input_size,hidden_size,output_size
        rng=random.Random(seed)
        self.w1=[[rng.uniform(-.18,.18) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1=[0.0]*hidden_size
        self.w2=[[rng.uniform(-.18,.18) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2=[0.0]*output_size; self.training_updates=0
    def _forward(self,x):
        if len(x)!=self.input_size: raise ValueError(f"Expected {self.input_size} input features, got {len(x)}")
        h=[math.tanh(self.b1[j]+sum(x[i]*self.w1[i][j] for i in range(self.input_size))) for j in range(self.hidden_size)]
        logits=[self.b2[k]+sum(h[j]*self.w2[j][k] for j in range(self.hidden_size)) for k in range(self.output_size)]
        return h,_softmax(logits)
    def predict(self,x): return self._forward(x)[1]
    def train_one(self,x,target_index,learning_rate=.08):
        if not 0<=target_index<self.output_size: raise ValueError("target_index is outside the output range")
        h,p=self._forward(x); od=p[:]; od[target_index]-=1.0; old_w2=[r[:] for r in self.w2]
        hd=[sum(od[k]*old_w2[j][k] for k in range(self.output_size))*(1-h[j]*h[j]) for j in range(self.hidden_size)]
        for j in range(self.hidden_size):
            for k in range(self.output_size): self.w2[j][k]-=learning_rate*h[j]*od[k]
        for k in range(self.output_size): self.b2[k]-=learning_rate*od[k]
        for i in range(self.input_size):
            for j in range(self.hidden_size): self.w1[i][j]-=learning_rate*x[i]*hd[j]
        for j in range(self.hidden_size): self.b1[j]-=learning_rate*hd[j]
        self.training_updates+=1
        return -math.log(max(p[target_index],1e-12))
    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        payload={"model_version":MODEL_VERSION,"input_size":self.input_size,"hidden_size":self.hidden_size,"output_size":self.output_size,"w1":self.w1,"b1":self.b1,"w2":self.w2,"b2":self.b2,"training_updates":self.training_updates}
        tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(payload),encoding="utf-8"); tmp.replace(path)
    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("model_version")!=MODEL_VERSION: raise ValueError("Unsupported model file version")
        m=cls(d["input_size"],d["hidden_size"],d["output_size"])
        if len(d["w1"])!=m.input_size or any(len(r)!=m.hidden_size for r in d["w1"]) or len(d["b1"])!=m.hidden_size or len(d["w2"])!=m.hidden_size or any(len(r)!=m.output_size for r in d["w2"]) or len(d["b2"])!=m.output_size: raise ValueError("Model file has invalid parameter dimensions")
        m.w1,m.b1,m.w2,m.b2=d["w1"],d["b1"],d["w2"],d["b2"]; m.training_updates=int(d.get("training_updates",0)); return m
