"""Build the dedicated V6 D2 + targeted D3-v2 run index."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from rtc.step2_data_index_v60 import build_step2_v60_run_index, v60_run_index_summary


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--d2-manifest",required=True); p.add_argument("--d2-run-summary",required=True); p.add_argument("--d3-v60-run-summary",required=True); p.add_argument("--out",required=True); a=p.parse_args()
    frame=build_step2_v60_run_index(d2_manifest=pd.read_csv(a.d2_manifest),d2_run_summary=pd.read_csv(a.d2_run_summary),d3_v60_run_summary=pd.read_csv(a.d3_v60_run_summary))
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(out,index=False); summary=v60_run_index_summary(frame); summary["out"]=str(out); print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__": main()
