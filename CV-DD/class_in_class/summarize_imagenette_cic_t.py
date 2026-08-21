import argparse,json
from pathlib import Path
from summarize_post_eval_seeds import hierarchical_summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--per-class-dir',required=True)
    p.add_argument('--recovery-seeds',nargs='+',type=int,required=True)
    p.add_argument('--student-seeds',nargs='+',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.per_class_dir);values={c:{} for c in (1,2,5,10)}
    for c in values:
        for r in a.recovery_seeds:
            for s in a.student_seeds:
                q=json.loads((root/f'c{c}_rseed{r}_sseed{s}.json').read_text())
                values[c][(r,s)]=float(q['best_top1'])
    summary={'protocol':'ImageNette IPC10 ResNet18 CiC-T only, marg10, T20',
             'recovery_seeds':a.recovery_seeds,'student_seeds':a.student_seeds,
             'arms':{f'C{c}':hierarchical_summary(values[c],a.recovery_seeds,a.student_seeds) for c in values},
             'paired_vs_C1':{}}
    for c in (2,5,10):
        d={k:values[c][k]-values[1][k] for k in values[c]}
        summary['paired_vs_C1'][f'C{c}_minus_C1']=hierarchical_summary(d,a.recovery_seeds,a.student_seeds)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(summary,indent=2);out.write_text(text+'\n');print(text)
if __name__=='__main__':main()
