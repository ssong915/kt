#! /bin/bash
DIM_SIZE=(128)
DATASET=('congress-bills' 'DAWN')
FREQ_SIZE=(1)
TEST_MODE=('train_fixed_split' 'train_live_update')
MODEL=('hgnn')
GPU=(1)

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