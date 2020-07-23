#!/usr/bin/env bash
mkdir -p results

# Reference Network training
for arch in resnetcif20 resnetcif32 resnetcif56 resnetcif110
do
python -u reference_trainer.py --arch ${arch} \
        --batch-size 128 --scheduler steps --milestones 100 150 \
        --lr 0.1 --lr_decay 0.1 --momentum 0.9 --epochs 200 \
        --checkpoint 20 --print-freq 5 \
        --dataset CIFAR10 | tee -a references/${arch}_reference.log
done


# Our compression
for exp_setup_name in resnet20_conv_only resnet32_conv_only resnet56_conv_only resnet110_conv_only
do
    for alpha in 1 2 4 8 16 32
    do
        alpha_exp="e-9"
        criterion="flops"
        epochs=15
        mu_init=1e-03
        mu_inc=1.25
        lc_steps=50
        type=lc
        lr=0.0007
        mu_rep=1
        lr_decay_mode=restart_on_l
        lr_decay=0.99
        momentum=0.9
        # we didn't use scheme_2 for CIFAR10 expts (only for ImageNet)
        conv_scheme=scheme_1
        tag=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 6 | head -n 1)
        logfile_lc=results/${exp_setup_name}_lc_${tag}_${criterion}_α=${alpha}${alpha_exp}.log
        # calling our compression
        python -u exp_runner.py \
                --exp_setup ${exp_setup_name} --type ${type} --tag ${tag} \
                --lc_steps ${lc_steps} --mu_init ${mu_init} --mu_inc ${mu_inc} --mu_rep ${mu_rep} \
                --lr ${lr} --lr_decay_mode  ${lr_decay_mode} --lr_decay ${lr_decay} --epochs ${epochs} \
                --momentum ${momentum} --alpha ${alpha}${alpha_exp} --criterion ${criterion} \
                --conv_scheme ${conv_scheme} | tee -a ${logfile_lc}


        ft_epochs=200
        type=ft
        lr=0.0001
        lr_decay_mode=restart_on_l
        lr_decay=0.99
        momentum=0.9
        # once compression is finished, we decompose the networks and then fine-tune
        logfile_ft="results/${exp_setup_name}_ft_${tag}.log"
        python -u exp_runner.py \
                --exp_setup ${exp_setup_name} --type ${type} --tag ${tag} \
                --lr ${lr}  --lr_decay ${lr_decay} --epochs ${ft_epochs} --momentum ${momentum} \
                | tee -a ${logfile_ft}
    done
done
