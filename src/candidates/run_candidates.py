"""CLI to create candidate tables and per-trip candidate coverage reports."""
import argparse
from pathlib import Path
from candidate_generation import build_partition

p=argparse.ArgumentParser()
p.add_argument("--root",required=True,help="Dataset root containing train/ and student_test/")
p.add_argument("--partition",choices=["train","student_test"],default="train")
p.add_argument("--k",type=int,default=64)
p.add_argument("--max-points",type=int,default=None,help="Optional deterministic sample per trip")
p.add_argument("--out",default="results")
a=p.parse_args()
out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
df,cov=build_partition(a.root,a.partition,a.k,a.max_points)
tag=f"{a.partition}_candidates_k{a.k}"+(f"_sample{a.max_points}" if a.max_points else "_full")
df.to_csv(out/f"{tag}.csv",index=False)
cov.to_csv(out/f"{a.partition}_coverage_k{a.k}.csv",index=False)
print("candidate rows:",len(df))
if "candidate_miss_rate" in cov and cov.candidate_miss_rate.notna().any():
    weighted=cov.candidate_misses.sum()/cov.points_used.sum()
    print("weighted candidate miss rate:",round(float(weighted),5))
    print("macro trip miss rate:",round(float(cov.candidate_miss_rate.mean()),5))
print(cov.sort_values("candidate_miss_rate",ascending=False,na_position="last").head(10).to_string(index=False))
