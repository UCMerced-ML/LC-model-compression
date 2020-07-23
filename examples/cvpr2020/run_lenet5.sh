#!/usr/bin/env bash
mkdir -p results

# Reference Network training
python -u reference_trainer.py --arch lenet5_drop \
        --batch-size 256 \
        --lr 0.1 --lr_decay 0.99 --momentum 0.9 --epochs 300 \
        --checkpoint 20 --print-freq 5 \
        --dataset MNIST

# Our compression
for alpha in 0.4 0.1 0.11 1.4 2 2.4 3
do
    exp_setup_name=lenet5_all
    alpha_exp="e-9"
    criterion="flops"
    epochs=30
    mu_init=1e-04
    mu_inc=1.1
    lc_steps=30
    type=lc
    lr=0.1
    mu_rep=1
    lr_decay_mode=after_l
    lr_decay=0.98
    momentum=0.9

    tag=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 6 | head -n 1)
    # calling our compression
    python -u exp_runner.py \
            --exp_setup ${exp_setup_name} --type ${type} --tag ${tag} \
            --lc_steps ${lc_steps} --mu_init ${mu_init} --mu_inc ${mu_inc} --mu_rep ${mu_rep} \
            --lr ${lr} --lr_decay_mode  ${lr_decay_mode} --lr_decay ${lr_decay} --epochs ${epochs} --momentum ${momentum} \
            --alpha ${alpha}${alpha_exp} --criterion ${criterion} --conv_scheme scheme_1 \
            | tee -a results/${exp_setup_name}_lc_${tag}_${criterion}_α=${alpha}${alpha_exp}.log


    ft_epochs=300
    type=ft
    lr=0.02
    lr_decay_mode=restart_on_l
    lr_decay=0.99
    # once compression is finished, we decompose the networks and then fine-tune
    logfile="results/${exp_setup_name}_ft_${tag}.log"
    python -u exp_runner.py \
            --exp_setup ${exp_setup_name} --type ${type} --tag ${tag} \
            --lr ${lr}  --lr_decay ${lr_decay} --epochs ${ft_epochs} --momentum ${momentum} \
            | tee -a ${logfile}
done
