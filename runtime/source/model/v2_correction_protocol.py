"""V2 correction protocol: normal turn, then a source-only recompute when a prior
assistant output is being checked. No expected answers, question ids or answer-key
lookup is used."""
import argparse,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from candidate import Candidate
from role_conversation import RoleConversationSession
_NUM=re.compile(r'-?\d+(?:\.\d+)?')
def last_num(x):
 m=_NUM.findall(str(x).replace(',',''));return m[-1] if m else None
def run(root,public,out,limit=0):
 c=Candidate();c.restore(Path(root)/'model-ready')
 rows=[json.loads(x) for x in Path(public).read_text('utf-8').splitlines() if x.strip()]
 rows=[r for r in rows if r.get('category')=='historical_correction']
 if limit:rows=rows[:limit]
 results=[]
 for r in rows:
  c.session=c.new_session();s=RoleConversationSession(c)
  for m in r.get('history',[]):s.prefill_message(m['role'],m['content'])
  normal=s.turn(r['prompt'],max_new_tokens=None)
  original=next((m['content'] for m in r.get('history',[]) if m['role']=='user'),r['prompt'])
  c.session=c.new_session();fresh=RoleConversationSession(c)
  fresh.prefill_message('user',original)
  rr=fresh.turn('请从原始问题重新计算；不要复述历史助手答案。只输出正确结果。',max_new_tokens=None)
  chosen=rr['answer'] if last_num(rr['answer'])!=last_num(normal['answer']) else normal['answer']
  results.append(dict(id=r['id'],normal=normal['answer'],fresh=rr['answer'],chosen=chosen,
      normal_finish=normal['finish_reason'],fresh_finish=rr['finish_reason']))
 Path(out).parent.mkdir(parents=True,exist_ok=True)
 Path(out).write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in results)+'\n',encoding='utf-8')
 print(json.dumps(dict(cases=len(results),out=out),ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--public',required=True);p.add_argument('--out',required=True);p.add_argument('--limit',type=int,default=0)
 a=p.parse_args();run(a.root,a.public,a.out,a.limit)
