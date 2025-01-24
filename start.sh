#! /bin/bash 

for model in "r2u_net" "fcn" "u_net" "nestedunet" "model" "u2a_net" "att_unet"
do
	for path in "/root/autodl-tmp/new_data" "/root/autodl-tmp/data2" "/root/autodl-tmp/data3" "root/autodl-tmp/data4"
		do
			python train.py --model $model --path $path
		done
done
	  