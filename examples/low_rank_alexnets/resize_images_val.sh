#!/usr/bin/env bash

target_val_folder=$1
cd $target_val_folder
for img in ./*.JPEG; do
    convert $img -resize "256x256^" $img
done