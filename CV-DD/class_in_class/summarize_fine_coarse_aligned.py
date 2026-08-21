import argparse, json
from pathlib import Path
from summarize_post_eval_seeds import hierarchical_summary, result_path

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-root',required=True)
    p.add_argument('--recovery-seeds',nargs='+',type=int,required=True)
    p.add_argument('--student-seeds',nargs='+',type=int,required=True)
    p.add_argument('--output-dir',required=True);a=p.parse_args()
    root=Path(a.experiment_root); new=root/'relabel_alignment_per_class'; old=root/'per_class'
    align=root/'relabel_alignment_per_class'; values={k:{} for k in ('fine_coarse_aligned','fine_coarse_existing','oracle_aligned')}
    for r in a.recovery_seeds:
        for s in a.student_seeds:
            paths={
                'fine_coarse_aligned':new/f'fine_coarse_aligned_rseed{r}_sseed{s}.json',
                'fine_coarse_existing':result_path(old,'fine_coarse_target',r,s,42,42),
                'oracle_aligned':align/f'oracle_aligned_rseed{r}_sseed{s}.json'}
            for name,path in paths.items():values[name][(r,s)]=float(json.loads(path.read_text())['best_top1'])
    summary={'arms':{k:hierarchical_summary(v,a.recovery_seeds,a.student_seeds) for k,v in values.items()},'paired_comparisons':{}}
    for name,pos,neg in (
        ('fine_coarse_aligned_minus_existing','fine_coarse_aligned','fine_coarse_existing'),
        ('oracle_aligned_minus_fine_coarse_aligned','oracle_aligned','fine_coarse_aligned')):
        d={key:values[pos][key]-values[neg][key] for key in values[pos]}
        summary['paired_comparisons'][name]=hierarchical_summary(d,a.recovery_seeds,a.student_seeds)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    text=json.dumps(summary,indent=2);(out/'summary.json').write_text(text+'\n');print(text)
if __name__=='__main__':main()
