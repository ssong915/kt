#! /bin/bash
DIM_SIZE=(128)
DATASET=('email-Enron' 'email-Eu' 'tags-ask-ubuntu' 'tags-math-sx' 'threads-ask-ubuntu' 'threads-math-sx')
FREQ_SIZE=(2628000)
TEST_MODE=('train_fixed_split' 'train_live_update')
MODEL=('hnhn')
GPU=(0)
for dataset_name in ${DATASET[*]}
do

for test_mode in ${TEST_MODE[*]}
do

for model in ${MODEL[*]}
do

for freq_size in ${FREQ_SIZE[*]}
do

for gpu in ${GPU[*]}
do

python3 -u main.py \
	--dataset_name=${dataset_name} \
	--model=${model} \
	--freq_size=${freq_size} \
	--test_mode=${test_mode} \
	--gpu=${gpu} \

&> logs/${dataset}.log

    sleep 5

done
done
done
done
done