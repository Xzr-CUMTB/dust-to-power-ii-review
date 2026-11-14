"""Joint latent Gaussian and normalized conditional affine-coupling flows. Dimensions time x site."""
import numpy as np, torch,joblib
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
class Encoder:
 def fit(self,x):
  self.s=StandardScaler().fit(x);self.p=PCA(n_components=min(14,len(x)-2,x.shape[1]),svd_solver='full').fit(self.s.transform(x));return self
 def __call__(self,x):return (self.p.transform(self.s.transform(x))/np.maximum(1,self.p.explained_variance_**.5)).clip(-5,5)
class Codec:
 def fit(self,y,mode,dt):
  self.mode,self.dt=mode,dt;self.h,self.s=y.shape[1:];self.nb=self.h//(self.h//(4 if self.h%4==0 else 3));self.bs=self.h//self.nb
  a,side=self.parts(y);self.p=PCA(n_components=min(18,len(y)-2,a.shape[1]),svd_solver='full').fit(a);self.side=side.shape[1]
  return self
 def parts(self,y):
  n=len(y);y=np.maximum(y,0)
  if self.mode!='D4':return np.sqrt(y).reshape(n,-1),np.zeros((n,0))
  a=y.reshape(n,self.nb,self.bs,self.s);b=a.sum(2)*self.dt;e=b.sum((1,2))
  logb=np.log(b+.35).reshape(n,-1);logb-=logb.mean(1,keepdims=True)
  shp=np.sqrt((a+.005)/(a.sum(2,keepdims=True)+.005*self.bs))
  return shp.reshape(n,-1),np.c_[np.log1p(e),logb[:,:-1]]
 def encode(self,y):
  a,b=self.parts(y);return np.column_stack([b,self.p.transform(a)])
 def decode(self,z,mask):
  a=self.p.inverse_transform(z[:,self.side:]).reshape(-1,self.h,self.s)
  if self.mode!='D4':return (np.maximum(a,0)**2)*mask
  e=np.expm1(z[:,0].clip(0,12));lb=z[:,1:self.side];lb=np.c_[lb,-lb.sum(1)].reshape(-1,self.nb,self.s).clip(-12,12)
  shape=np.maximum(a.reshape(-1,self.nb,self.bs,self.s),.002)**2*mask.reshape(-1,self.nb,self.bs,self.s)
  mass=np.exp(lb)* (shape.sum(2)>0);mass/=np.maximum(mass.sum((1,2),keepdims=True),1e-20)
  b=np.maximum(0,(e+.35*(self.nb*self.s))[:,None,None]*mass-.35);b/=np.maximum(b.sum((1,2),keepdims=True),1.e-12)
  path=(shape/np.maximum(shape.sum(2,keepdims=True),1e-20)*b[:,:,None,:]*e[:,None,None,None]/self.dt).reshape(-1,self.h,self.s)
  assert np.allclose(path.sum((1,2))*self.dt,e,atol=.02) or np.any(~mask.any((1,2)))
  return path
class Flow(nn.Module):
 def __init__(self,d,c):
  super().__init__();self.nets=nn.ModuleList();self.d=d
  for i in range(4):
   net=nn.Sequential(nn.Linear(d+c,48),nn.SiLU(),nn.Linear(48,2*d));nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias);self.nets.append(net)
 def move(self,z,c,reverse=False):
  det=0.
  for i in (reversed(range(4)) if reverse else range(4)):
   mask=((torch.arange(self.d,device=z.device)+i)%2).float();u=z*mask;t,ls=self.nets[i](torch.cat([u,c],-1)).chunk(2,-1)
   ls=.65*torch.tanh(ls)*(1-mask);t=t*(1-mask)
   z=(z-t)*torch.exp(-ls) if reverse else z*torch.exp(ls)+t
   det=det+ls.sum(-1)
  return z,det
def fit(x,y,fitix,valix,kind,stem,seed=42,r2=False,dt=.25):
 torch.manual_seed(seed);torch.set_num_threads(1);rng=np.random.default_rng(seed);enc=Encoder().fit(x[fitix]);a=enc(x);codec=Codec().fit(y[fitix],kind,dt)
 z=codec.encode(y);reg=Ridge(alpha=80).fit(a[fitix],z[fitix]);err=z-reg.predict(a)
 sd=np.maximum(np.std(err[fitix],0),.025);v=err/sd
 cov=np.atleast_2d(np.cov(v[fitix].T));L=np.linalg.cholesky(.84*cov+.16*np.eye(len(sd)));v=np.linalg.solve(L,v.T).T
 meta=dict(enc=enc,codec=codec,reg=reg,sd=sd,L=L,r2=r2,kind=kind);joblib.dump(meta,str(stem)+'.joblib')
 if kind=='D2':return []
 device='cuda';cf=torch.tensor(a,dtype=torch.float32,device=device);zz=torch.tensor(v,dtype=torch.float32,device=device);net=Flow(z.shape[1],a.shape[1]).to(device)
 opt=torch.optim.AdamW(net.parameters(),lr=.0012,weight_decay=.003);best=np.inf;logs=[]
 # Optional second-stage proper quantile auxiliary targets the full marginal of log day energy.
 fixed=torch.randn(12,len(fitix),z.shape[1],device=device);cl=torch.tensor(L[0]*sd[0],dtype=torch.float32,device=device)
 for ep in range(1,101):
  net.train()
  for j in np.array_split(rng.permutation(len(fitix)),max(1,len(fitix)//56)):
   ix=fitix[j];u,det=net.move(zz[ix],cf[ix],True);loss=(.5*u.square().mean(-1)+det/z.shape[1]).mean()
   if r2:
    # jointly train quantile of energy residual and base joint density (no reweighting observed labels).
    c=cf[ix].expand(12,-1,-1);draw,_=net.move(fixed[:,j],c);en=draw@cl;obs=zz[ix]@cl
    delta=obs-en.quantile(.15,dim=0);loss=loss+.7*torch.maximum(.15*delta,-.85*delta).mean()
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
  with torch.no_grad():
   u,dd=net.move(zz[valix],cf[valix],True);vl=float((.5*u.square().mean(-1)+dd/z.shape[1]).mean())
  logs.append([ep,float(loss.detach()),vl])
  if vl<best-.0002:
   best=vl;last=ep;torch.save(dict(state=net.state_dict(),d=z.shape[1],c=a.shape[1],epoch=ep),str(stem)+'.pt')
  if ep-last>=16:break
 return logs
def sample(stem,x,mask,seed=851,S=32):
 m=joblib.load(str(stem)+'.joblib');a=m['enc'](x);med=m['reg'].predict(a)
 n,d=med.shape;r=np.random.default_rng(seed).normal(size=(n,S,d)).astype('f4')
 if m['kind']!='D2':
  z=torch.load(str(stem)+'.pt',weights_only=True,map_location='cpu');net=Flow(z['d'],z['c']);net.load_state_dict(z['state']);net.eval()
  with torch.no_grad():r=net.move(torch.from_numpy(r),torch.tensor(np.repeat(a[:,None],S,1),dtype=torch.float32))[0].numpy()
 score=med[:,None]+(r@m['L'].T)*m['sd'];co=m['codec'];return co.decode(score.reshape(-1,d),np.repeat(mask,S,axis=0)).reshape(n,S,co.h,co.s).astype('f4')

