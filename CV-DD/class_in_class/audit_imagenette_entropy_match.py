import argparse, concurrent.futures, json, math, statistics
from pathlib import Path
from audit_imagenette_consumed_fkd_labels import analyze_root, summarize_roots

def main():
    p=argparse.ArgumentParser();p.add_argument('--random-root',required=True);p.add_argument('--factorial-root',required=True);p.add_argument('--ipc-root',required=True);p.add_argument('--match-root',required=True);p.add_argument('--c1-temperature',type=float,required=True);p.add_argument('--r100-temperature',type=float,required=True);p.add_argument('--teacher-seeds',nargs='+',type=int,default=(43,44));p.add_argument('--recovery-seeds',nargs='+',type=int,default=(41,42,43));p.add_argument('--epoch-stride',type=int,default=10);p.add_argument('--workers',type=int,default=8);p.add_argument('--output',required=True);a=p.parse_args()
    rr,fr,ir,mr=map(Path,(a.random_root,a.factorial_root,a.ipc_root,a.match_root)); tasks=[]
    def source(ipc,row,t,r):
        if ipc==1:return ir/f'tseed{t}'/'sources'/f'{row}_ipc1_rseed{r}'
        return fr/'real_sets'/f'tseed{t}_rseed{r}' if row=='real' else rr/f'tseed{t}'/'synthetic'/f'cic_t_c1_ipc10_rseed{r}'
    def ref_fkd(ipc,row,col,t,r):
        if ipc==1:return ir/f'tseed{t}'/'fkd'/f'ipc1_{row}__{col}_rseed{r}_bs10_ipc1'
        if row=='real':return fr/f'tseed{t}'/'fkd'/f'real__{col}_rseed{r}_bs10_ipc10'
        return rr/f'tseed{t}'/'fkd'/f'cic_t_c1_rseed{r}_bs10_ipc10' if col=='c1' else fr/f'tseed{t}'/'fkd'/f'c1__random100_rseed{r}_bs10_ipc10'
    tags={'c1':f'c1_T{str(a.c1_temperature).replace(".","p")}','random100':f'random100_T{str(a.r100_temperature).replace(".","p")}'}
    for ipc in (1,10):
      for row in ('real','c1'):
       for col,temp in (('c1',a.c1_temperature),('random100',a.r100_temperature)):
        for t in a.teacher_seeds:
         for r in a.recovery_seeds:
          src=source(ipc,row,t,r); tasks.append((f'adjusted_ipc{ipc}_{row}_{col}',100,t,r,str(src),str(mr/f'tseed{t}'/'fkd'/f'ipc{ipc}_{row}__{tags[col]}_rseed{r}_bs10_ipc{ipc}'),300,a.epoch_stride,temp,42));tasks.append((f'reference_ipc{ipc}_{row}_{col}',100,t,r,str(src),str(ref_fkd(ipc,row,col,t,r)),300,a.epoch_stride,20.0,42))
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex: rows=list(ex.map(analyze_root,tasks))
    summaries={}; matches={}; logk=math.log(10)
    for ipc in (1,10):
      for row in ('real','c1'):
       for kind in ('adjusted','reference'):
        for col in ('c1','random100'):
         key=f'{kind}_ipc{ipc}_{row}_{col}'; summaries[key]=summarize_roots([x for x in rows if x['partition']==key],a.teacher_seeds,a.recovery_seeds)
       hc=summaries[f'adjusted_ipc{ipc}_{row}_c1']['entropy']['mean_across_teacher_recovery_roots']; hr=summaries[f'adjusted_ipc{ipc}_{row}_random100']['entropy']['mean_across_teacher_recovery_roots']; tc=summaries[f'reference_ipc{ipc}_{row}_random100']['entropy']['mean_across_teacher_recovery_roots']; tr=summaries[f'reference_ipc{ipc}_{row}_c1']['entropy']['mean_across_teacher_recovery_roots']; matches[f'ipc{ipc}_{row}']={'c1_adjusted_entropy':hc,'target_random_T20_entropy':tc,'c1_minus_target':hc-tc,'random_adjusted_entropy':hr,'target_c1_T20_entropy':tr,'random_minus_target':hr-tr,'suggested_c1_temperature':a.c1_temperature*math.sqrt(max(logk-hc,1e-12)/max(logk-tc,1e-12)),'suggested_random_temperature':a.r100_temperature*math.sqrt(max(logk-hr,1e-12)/max(logk-tr,1e-12))}
    result={'temperatures':{'c1':a.c1_temperature,'random100':a.r100_temperature},'matches':matches,'summaries':summaries};out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
