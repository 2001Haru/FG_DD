import argparse,json
from pathlib import Path
from summarize_imagenette_cic_t_teacher_seeds import three_level_summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--random-root',required=True);p.add_argument('--factorial-root',required=True);p.add_argument('--ipc-root',required=True);p.add_argument('--match-root',required=True);p.add_argument('--c1-temperature',type=float,required=True);p.add_argument('--r100-temperature',type=float,required=True);p.add_argument('--output',required=True);a=p.parse_args();rr,fr,ir,mr=map(Path,(a.random_root,a.factorial_root,a.ipc_root,a.match_root));ts=(43,44);rs=(41,42,43);ss=(42,43,44);tags={'c1':f'c1_T{str(a.c1_temperature).replace(".","p")}','random100':f'random100_T{str(a.r100_temperature).replace(".","p")}' }
 def ref(ipc,row,col,t,r,s):
  if ipc==1:return ir/f'tseed{t}'/'per_class'/f'ipc1_{row}__{col}_rseed{r}_sseed{s}.json'
  if row=='real':return fr/f'tseed{t}'/'per_class'/f'real__{col}_rseed{r}_sseed{s}.json'
  if col=='c1':return rr/f'tseed{t}'/'per_class'/f'c1_rseed{r}_sseed{s}.json'
  return fr/f'tseed{t}'/'per_class'/f'c1__random100_rseed{r}_sseed{s}.json'
 out={}
 for ipc in (1,10):
  for row in ('real','c1'):
   for col in ('c1','random100'):
    adj={};base={};other={};othercol='random100' if col=='c1' else 'c1'
    for t in ts:
     for r in rs:
      for s in ss:
       key=(t,r,s);adj[key]=json.loads((mr/f'tseed{t}'/'per_class'/f'ipc{ipc}_{row}__{tags[col]}_rseed{r}_sseed{s}.json').read_text())['best_top1'];base[key]=json.loads(ref(ipc,row,col,t,r,s).read_text())['best_top1'];other[key]=json.loads(ref(ipc,row,othercol,t,r,s).read_text())['best_top1']
    out[f'ipc{ipc}_{row}_{col}']={'adjusted':three_level_summary(adj,ts,rs,ss),'original_same_teacher_T20':three_level_summary(base,ts,rs,ss),'entropy_target_other_teacher_T20':three_level_summary(other,ts,rs,ss),'adjusted_minus_original':three_level_summary({k:adj[k]-base[k] for k in adj},ts,rs,ss),'adjusted_minus_entropy_target_teacher':three_level_summary({k:adj[k]-other[k] for k in adj},ts,rs,ss)}
 result={'temperatures':{'c1':a.c1_temperature,'random100':a.r100_temperature},'cells':out};o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
