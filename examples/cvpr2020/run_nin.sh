#!/usr/bin/env bash
mkdir -p results

python -u reference_trainer.py --arch nincif_bn \
        --batch-size 128 --scheduler steps --milestones 350 600 700 800 \
        --lr 0.05 --lr_decay 0.1 --momentum 0.9 --epochs 900 \
        --checkpoint 20 --print-freq 5 \
        --dataset CIFAR10 | tee -a logs/nincif_bn_reference.log

# Our compression
for alpha in 0.025 0.05 0.075 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.9 1.3 1.4 2 3 3.5
do
    alpha_exp="e-9"
    criterion="flops"
    exp_setup_name=nin_all
    epochs=20
    mu_init=2e-05
    mu_inc=1.2
    lc_steps=60
    type=lc
    lr=0.0007
    mu_rep=1
    lr_decay_mode=restart_on_l
    lr_decay=0.99
    momentum=0.9

    tag=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 6 | head -n 1)
    python -u exp_runner.py \
            --exp_setup ${exp_setup_name} --type ${type} --tag ${tag} \
            --lc_steps ${lc_steps} --mu_init ${mu_init} --mu_inc ${mu_inc} --mu_rep ${mu_rep} \
            --lr ${lr} --lr_decay_mode  ${lr_decay_mode} --lr_decay ${lr_decay} --epochs ${epochs} --momentum ${momentum} \
            --alpha ${alpha}${alpha_exp} --criterion ${criterion} --conv_scheme scheme_1 | tee -a results/${exp_setup_name}_lc_${tag}_${criterion}_α=${alpha}${alpha_exp}.log


    ft_epochs=500
    type=ft
    lr=0.0005
    lr_decay_mode=restart_on_l
    lr_decay=0.99
    momentum=0.9
    # once compression is finished, we decompose the networks and then fine-tune
    logfile="results/${exp_setup_name}_ft_${tag}.log"
    python -u exp_runner.py \
            --exp_setup ${exp_setup_name} --type ${type} --tag ${tag} \
            --lr ${lr}  --lr_decay ${lr_decay} --epochs ${ft_epochs} --momentum ${momentum} \
            | tee -a ${logfile}
done