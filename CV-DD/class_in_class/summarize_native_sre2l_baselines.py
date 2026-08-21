import argparse,json
from pathlib import Path
from summarize_post_eval_seeds import hierarchical_summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--per-class-dir',required=True)
    p.add_argument('--recovery-seeds',nargs='+',type=int,required=True)
    p.add_argument('--student-seeds',nargs='+',type=int,required=True)
    p.add_argument('--output-dir',required=True);a=p.parse_args();root=Path(a.per_class_dir)
    values={10:{},50:{}}
    for ipc in values:
        for r in a.recovery_seeds:
            for s in a.student_seeds:
                path=root/f'ipc{ipc}_rseed{r}_sseed{s}.json'
                values[ipc][(r,s)]=float(json.loads(path.read_text())['best_top1'])
    summary={'protocol':'native CV-DD SRe2L++ CIFAR100 100-way, T=20, 300 epochs',
             'recovery_seeds':a.recovery_seeds,'student_seeds':a.student_seeds,
             'ipc10':hierarchical_summary(values[10],a.recovery_seeds,a.student_seeds),
             'ipc50':hierarchical_summary(values[50],a.recovery_seeds,a.student_seeds)}
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    text=json.dumps(summary,indent=2);(out/'summary.json').write_text(text+'\n');print(text)
if __name__=='__main__':main()
