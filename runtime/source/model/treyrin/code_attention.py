"""Token-address attention over the core's retained physical episode slots.

Fixed hashed unigram/bigram addresses supply detail lost by the 32D semantic
key. Learned cortical read mass is the prior. Both branches inspect only the
current public question and admitted model-state token codes; no journal,
private answer, record-name parsing or external retriever participates.
This layer is parameter-free; it does not claim learned semantic compression.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TokenCodeAttention(nn.Module):
    def __init__(self,features=8192,gain=128.,coverage=.05,block_size=256,hops=1):
        super().__init__()
        if type(features)is not int or features<64 or features&(features-1):raise ValueError('Power-of-two token address space required')
        if not math.isfinite(gain)or gain<=0 or not 0<coverage<1:raise ValueError('Finite attention scale/coverage required')
        if type(block_size)is not int or block_size<1:raise ValueError('Positive bounded address block size required')
        self.features=features;self.gain=gain;self.coverage=coverage;self.block_size=block_size
        if type(hops)is not int or hops not in(1,2):raise ValueError('One or two internal attention hops required')
        self.hops=hops

    def features_for(self,codes,lengths):
        if codes.ndim!=2 or lengths.shape!=codes.shape[:1]:raise ValueError('Aligned physical codes and lengths required')
        if codes.dtype not in(torch.int32,torch.int64):raise ValueError('Integer model token codes required')
        n,t=codes.shape;x=codes.long();valid=torch.arange(t,device=x.device)[None]<lengths[:,None]
        counts=torch.zeros(n,self.features,device=x.device,dtype=torch.float32)
        h=(x*1103515245+12345)&(self.features-1)
        counts.scatter_add_(1,h,valid.to(counts.dtype)*.25)
        if t>1:
            h=(x[:,:-1]*214013+x[:,1:]*2531011+137)&(self.features-1)
            counts.scatter_add_(1,h,(valid[:,:-1]&valid[:,1:]).to(counts.dtype))
        return counts.log1p()

    def forward(self,query_ids,codes,lengths,semantic_attention,allocation):
        if query_ids.ndim!=1 or query_ids.numel()<1:raise ValueError('One public question token sequence required')
        if semantic_attention.ndim!=2 or semantic_attention.shape[1]!=codes.shape[0]or allocation.shape!=(semantic_attention.shape[0]+1,):raise ValueError('Core read surfaces differ from physical token slots')
        if not torch.isfinite(semantic_attention).all()or not torch.isfinite(allocation).all()or bool((semantic_attention<0).any())or bool((allocation<0).any()):
            raise ValueError('Finite nonnegative model read probabilities required')
        occupied=lengths>0;n=int(occupied.sum());count=len(lengths)
        if not n:
            p=torch.zeros(count+1,device=codes.device);p[-1]=1
            return dict(probabilities=p,abstained=True,occupied=0)
        q=self.features_for(query_ids[None],torch.tensor([len(query_ids)],device=codes.device))
        # Features shared by every retained item (roles, source framing, common
        # noise) have zero IDF and cannot become a content shortcut.
        indices=occupied.nonzero().flatten()
        df=torch.zeros(self.features,device=codes.device)
        # Two bounded passes avoid materialising a 16k x 8192 address matrix
        # on the 8 GiB deployment GPU. Integer document counts stay exact.
        for block in indices.split(self.block_size):
            f=self.features_for(codes[block],lengths[block]);df+=(f>0).sum(0)
        idf=math.log1p(n)-df.log1p();q=F.normalize(q*idf,dim=-1)
        cosines=[]
        for block in indices.split(self.block_size):
            f=self.features_for(codes[block],lengths[block]);f=F.normalize(f*idf,dim=-1)
            cosines.append((f*q).sum(-1))
        cosine=torch.cat(cosines)
        prior=(semantic_attention.float()*allocation[:-1,None].float()).sum(0)[occupied]
        prior=(1-self.coverage)*prior+self.coverage/n
        scores=prior.clamp_min(1e-30).log()+self.gain*cosine
        association_strength=scores.new_tensor(0.)
        if self.hops==2 and n>1:
            first=scores.softmax(-1)
            concentration=(1+(first*first.clamp_min(1e-30).log()).sum()/math.log(n)).clamp(0,1)
            # Spread activation from actual first-read codes, using only
            # features shared by multiple retained episodes. Unique record
            # IDs cannot become associative links to another record.
            chosen=first.topk(min(2,n)).indices;mass=first[chosen]/first[chosen].sum()
            f=self.features_for(codes[indices[chosen]],lengths[indices[chosen]])*idf*(df>=2)
            association=F.normalize((F.normalize(f,dim=-1)*mass[:,None]).sum(0,keepdim=True),dim=-1)
            association_strength=.35*concentration
            refined=F.normalize(q+association_strength*association,dim=-1)
            cosines=[]
            for block in indices.split(self.block_size):
                f=self.features_for(codes[block],lengths[block]);f=F.normalize(f*idf,dim=-1)
                cosines.append((f*refined).sum(-1))
            cosine=torch.cat(cosines);scores=prior.clamp_min(1e-30).log()+self.gain*cosine
        logits=torch.full((count+1,),float('-inf'),device=codes.device)
        logits[:-1][occupied]=scores
        logits[-1]=allocation[-1].float().clamp_min(1e-30).log()
        p=logits.softmax(-1)
        return dict(probabilities=p,abstained=int(logits.argmax())==count,
                    occupied=n,cosine=cosine.detach(),physical_indices=occupied.nonzero().flatten(),
                    hops=self.hops,association_strength=float(association_strength.detach()))
