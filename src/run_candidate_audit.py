"""Legacy entry point; delegates to the current candidate pipeline."""
import argparse
from pathlib import Path
from candidates.candidate_generation import build_partition

p=argparse.ArgumentParser()
p.add_argument("--root",required=True)
p.add_argument("--partition",default="train",choices=["train","student_test"])
p.add_argument("--k",type=int,default=64)
p.add_argument("--max-points",type=int,default=None)
p.add_argument("--out",default="results")
a=p.parse_args()
out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
df,cov=build_partition(a.root,a.partition,a.k,a.max_points)
df.to_csv(out/f"{a.partition}_candidates.csv",index=False)
cov.to_csv(out/f"{a.partition}_coverage.csv",index=False)
print("candidate rows:",len(df))
print(cov.to_string(index=False))
