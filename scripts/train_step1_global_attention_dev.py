"""Stable Development-only trainer for the V122 Step1 global-attention ablation.

This script does not replace the frozen production Step1.  Train a separate attention
checkpoint, compare it with ``audit_step1_global_attention_current.py`` on identical held-out
Development windows, and only then decide whether a new Step1/state-store lineage is justified.
"""
from rtc.step1_train_v122 import train_step1_v122_main


if __name__ == "__main__":
    train_step1_v122_main()
