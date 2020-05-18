#!/usr/bin/env bash
# we will be writing all outputs to log files (and showing in stdout) using tee
mkdir -p logs
for exp_name in pruning quantize_all quantize_two_layers all_mixed low_rank low_rank_with_selection additive_quant_and_prune; do
    python -u lenet300.py --exp_name $exp_name | tee -a logs/"$exp_name".txt
done