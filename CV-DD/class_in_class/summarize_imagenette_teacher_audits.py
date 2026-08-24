import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit-dir',required=True);p.add_argument('--output',required=True)
    p.add_argument('--c-values',nargs='+',type=int,default=(1,2,5,10))
    p.add_argument('--file-template',default='random_c{c}_teacher_audit.json')
    p.add_argument('--partition-description',default='balanced random within each ImageNette parent class')
    a=p.parse_args()
    root=Path(a.audit_dir); rows=[]
    for c in a.c_values:
        q=json.loads((root/a.file_template.format(c=c)).read_text())
        rows.append({'C':c,'heads':10*c,'expected_test_ratio':1/c,
                     'train_native_top1':q['train']['native_subclass_top1'],
                     'train_coarse_top1':q['train']['collapsed_coarse10_top1'],
                     'train_entropy':q['train']['within_parent_entropy'],
                     'val_native_top1':q['val']['native_subclass_top1'],
                     'val_coarse_top1':q['val']['collapsed_coarse10_top1'],
                     'val_native_to_coarse_ratio':q['val']['native_to_collapsed_hit_ratio'],
                     'val_conditional_native_given_coarse':q['val']['conditional_native_given_coarse_correct'],
                     'val_conditional_binomial_test':q['val']['conditional_ratio_binomial_test'],
                     'val_entropy':q['val']['within_parent_entropy']})
    result={'partition':a.partition_description,'teachers':rows}
    text=json.dumps(result,indent=2);out=Path(a.output);out.write_text(text+'\n');print(text)
if __name__=='__main__':main()
