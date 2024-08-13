#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1
#SBATCH -w aiwkr2


#module load cuda/11.7.0
#module load singularity/3.11.0
module load cuda/11.8.0
module load anaconda/anaconda3-2022.10

source activate py310t2cu118

# ele exchange weather的全连接消融       2024.7.8 08：19

use_x_mark_enc=0
front_use_decomp=0
use_space_merge=0


use_fourier=1
use_conv=1


seq_len=96
if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log
fi
if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55
fi
if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/ETTm1" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/ETTm1
fi
if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/ETTh1" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/ETTh1
fi
if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/ETTm2" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/ETTm2
fi

if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/ETTh2" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/ETTh2
fi
if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/electricity" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/electricity
fi

if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/Exchange" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/Exchange
fi

if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/Solar" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/Solar
fi

if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/weather" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/weather
fi

if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/Traffic" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/Traffic
fi

if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/PEMS03" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/PEMS03
fi

if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/PEMS04" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/PEMS04
fi

if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/PEMS07" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/PEMS07
fi
if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/PEMS08" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/PEMS08
fi
#singularity exec --nv /mnt/nfs/data/home/1120231455/home/fuy/fuypycharm1_1.sif  nvidia-smi;\

date_ETTh1=ETTh1
date_ETTh2=ETTh2
date_ETTm1=ETTm1
date_ETTm2=ETTm2
date_exchange_rate=exchange_rate
date_national_illness=national_illness
date_weather=weather
# 分解核除了ill是 19 13，其他都是13 17  且必须为奇数
decomp_kernel=(17 49 81)
electricity_x_mark_len=4
weather_x_mark_len=5
Exchange_x_mark_len=3
illness_x_mark_len=3
traffic_x_mark_len=4
ETTh1_x_mark_len=4
ETTh2_x_mark_len=4
ETTm2_x_mark_len=5
ETTm1_x_mark_len=5

Solar_x_mark_len=1
PEMS03_x_mark_len=1
PEMS04_x_mark_len=1
PEMS07_x_mark_len=1
PEMS08_x_mark_len=1













# 原始序列1 + 全连接1 + relu1
use_origin_seq=1
use_linear=1
use_relu=1

#  h1 2分解核  dmodel=dff=256
for down_sampling_layer in 3 4 5 6 7 8 9 10; do

  for pred_len in 96  192 336 720; do
     python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id TaperFourierFormer'_ETTh1_'96'_'$pred_len \
        --root_path ETT-small \
        --mode regre \
        --data ETTh1 \
        --freq h \
        --features M \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 256 \
        --d_ff 256 \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len $ETTh1_x_mark_len \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_origin_seq $use_origin_seq \
        --use_linear $use_linear \
        --use_relu $use_relu \
      > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/ETTh1/'0_TaperFourierFormer_'ETTh1'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done

done


#  h2 2分解核  d_model= d_ff = 128
for down_sampling_layer in 3 4 5 6 7 8 9 10; do

for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --root_path ETT-small \
        --model_id TaperFourierFormer'_ETTh2_'96'_'$pred_len \
        --mode regre \
        --data ETTh2 \
        --freq h \
        --features M \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len $ETTh2_x_mark_len \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_origin_seq $use_origin_seq \
        --use_linear $use_linear \
        --use_relu $use_relu \
      > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/ETTh2/'0_TaperFourierFormer_'ETTh1'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done

done


# m1 d_model= d_ff = 128
for down_sampling_layer in 3 4 5 6 7 8 9 10; do

  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id TaperFourierFormer'_ETTm1_'96'_'$pred_len \
        --root_path ETT-small \
        --mode regre \
        --freq t \
        --data ETTm1 \
        --features M \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len $ETTm1_x_mark_len \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_origin_seq $use_origin_seq \
        --use_linear $use_linear \
        --use_relu $use_relu \
      > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/ETTm1/'0_TaperFourierFormer_'ETTh1'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
done


#  m2 2分解核

# m2 d_model= d_ff = 128
for down_sampling_layer in 3 4 5 6 7 8 9 10; do

  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --root_path ETT-small \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id TaperFourierFormer'_ETTm2_'96'_'$pred_len \
        --model TaperFourierFormer \
        --mode regre \
        --data ETTm2 \
        --features M \
        --d_layers 1 \
        --freq t \
        --seq_len 96 \
        --pred_len $pred_len \
        --d_model 128 \
        --d_ff 128 \
        --x_enc_len 7 \
        --x_mark_len $ETTm2_x_mark_len \
        --e_layers 2 \
        --itr 1 \
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_origin_seq $use_origin_seq \
        --use_linear $use_linear \
        --use_relu $use_relu \
      > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/ETTm2/'0_TaperFourierFormer_'ETTh1'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
done

if false;then

# electricity d_model=d_ff=512
for down_sampling_layer in 3 4 5 6 7 8 9 10; do

 for pred_len in 96 192 336 720 ; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id TaperFourierFormer'_ECL_'96'_'$pred_len \
        --root_path electricity \
        --mode regre \
        --data ECL \
        --freq h \
        --features M \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 512 \
        --d_ff 512 \
        --batch_size 16 \
        --seq_len 96 \
        --pred_len $pred_len \
        --itr 1 \
        --learning_rate 0.0005 \
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $electricity_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_origin_seq $use_origin_seq \
        --use_linear $use_linear \
        --use_relu $use_relu \
      > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/electricity/'0_TaperFourierFormer_'ETTh1'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
done

fi



# Exchange d_model=d_ff=128
for down_sampling_layer in 3 4 5 6 7 8 9 10; do

  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id TaperFourierFormer'_Exchange_'96'_'$pred_len \
        --model TaperFourierFormer \
        --root_path exchange_rate \
        --mode regre \
        --data Exchange \
        --features M \
        --e_layers 2 \
        --freq d \
        --d_layers 1 \
        --d_model 128 \
        --d_ff 128 \
        --seq_len 96 \
        --itr 1 \
        --pred_len $pred_len \
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $Exchange_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_origin_seq $use_origin_seq \
        --use_linear $use_linear \
        --use_relu $use_relu \
      > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/Exchange/'0_TaperFourierFormer_'ETTh1'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done

done


if false;then
#  weather 3分解核
# weather d_ff=d_ff=512
for down_sampling_layer in 3 4 5 6 7 8 9 10; do

  for pred_len in 96 192 336 720 ; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id TaperFourierFormer'_WTH_'96'_'$pred_len \
        --model TaperFourierFormer \
        --root_path weather \
        --mode regre \
        --data WTH \
        --features M \
        --freq t \
        --d_layers 1 \
        --e_layers 2 \
        --d_ff 512 \
        --d_model 512 \
        --batch_size 16 \
        --seq_len 96 \
        --pred_len $pred_len \
        --itr 1 \
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $weather_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_origin_seq $use_origin_seq \
        --use_linear $use_linear \
        --use_relu $use_relu \
      > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/weather/'0_TaperFourierFormer_'ETTh1'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
done

fi






#  h1 2分解核  dmodel=dff=256
for down_sampling_layer in 0 1 2 11 128; do

  for pred_len in 96  192 336 720; do
     python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id TaperFourierFormer'_ETTh1_'96'_'$pred_len \
        --root_path ETT-small \
        --mode regre \
        --data ETTh1 \
        --freq h \
        --features M \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 256 \
        --d_ff 256 \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len $ETTh1_x_mark_len \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_origin_seq $use_origin_seq \
        --use_linear $use_linear \
        --use_relu $use_relu \
      > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/ETTh1/'0_TaperFourierFormer_'ETTh1'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done

done


#  h2 2分解核  d_model= d_ff = 128
for down_sampling_layer in 0 1 2 11 128; do

for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --root_path ETT-small \
        --model_id TaperFourierFormer'_ETTh2_'96'_'$pred_len \
        --mode regre \
        --data ETTh2 \
        --freq h \
        --features M \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len $ETTh2_x_mark_len \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_origin_seq $use_origin_seq \
        --use_linear $use_linear \
        --use_relu $use_relu \
      > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/ETTh2/'0_TaperFourierFormer_'ETTh1'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done

done


# m1 d_model= d_ff = 128
for down_sampling_layer in 0 1 2 11 128; do

  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id TaperFourierFormer'_ETTm1_'96'_'$pred_len \
        --root_path ETT-small \
        --mode regre \
        --freq t \
        --data ETTm1 \
        --features M \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len $ETTm1_x_mark_len \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_origin_seq $use_origin_seq \
        --use_linear $use_linear \
        --use_relu $use_relu \
      > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/ETTm1/'0_TaperFourierFormer_'ETTh1'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
done


#  m2 2分解核

# m2 d_model= d_ff = 128
for down_sampling_layer in 0 1 2 11 128; do

  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --root_path ETT-small \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id TaperFourierFormer'_ETTm2_'96'_'$pred_len \
        --model TaperFourierFormer \
        --mode regre \
        --data ETTm2 \
        --features M \
        --d_layers 1 \
        --freq t \
        --seq_len 96 \
        --pred_len $pred_len \
        --d_model 128 \
        --d_ff 128 \
        --x_enc_len 7 \
        --x_mark_len $ETTm2_x_mark_len \
        --e_layers 2 \
        --itr 1 \
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_origin_seq $use_origin_seq \
        --use_linear $use_linear \
        --use_relu $use_relu \
      > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/ETTm2/'0_TaperFourierFormer_'ETTh1'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
done

if false;then

# electricity d_model=d_ff=512
for down_sampling_layer in 3 4 5 6 7 8 9 10; do

 for pred_len in 96 192 336 720 ; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id TaperFourierFormer'_ECL_'96'_'$pred_len \
        --root_path electricity \
        --mode regre \
        --data ECL \
        --freq h \
        --features M \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 512 \
        --d_ff 512 \
        --batch_size 16 \
        --seq_len 96 \
        --pred_len $pred_len \
        --itr 1 \
        --learning_rate 0.0005 \
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $electricity_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_origin_seq $use_origin_seq \
        --use_linear $use_linear \
        --use_relu $use_relu \
      > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/electricity/'0_TaperFourierFormer_'ETTh1'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
done

fi



# Exchange d_model=d_ff=128
for down_sampling_layer in 0 1 2 11 128; do

  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id TaperFourierFormer'_Exchange_'96'_'$pred_len \
        --model TaperFourierFormer \
        --root_path exchange_rate \
        --mode regre \
        --data Exchange \
        --features M \
        --e_layers 2 \
        --freq d \
        --d_layers 1 \
        --d_model 128 \
        --d_ff 128 \
        --seq_len 96 \
        --itr 1 \
        --pred_len $pred_len \
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $Exchange_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_origin_seq $use_origin_seq \
        --use_linear $use_linear \
        --use_relu $use_relu \
      > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/Exchange/'0_TaperFourierFormer_'ETTh1'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done

done


if false;then
#  weather 3分解核
# weather d_ff=d_ff=512
for down_sampling_layer in 3 4 5 6 7 8 9 10; do

  for pred_len in 96 192 336 720 ; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id TaperFourierFormer'_WTH_'96'_'$pred_len \
        --model TaperFourierFormer \
        --root_path weather \
        --mode regre \
        --data WTH \
        --features M \
        --freq t \
        --d_layers 1 \
        --e_layers 2 \
        --d_ff 512 \
        --d_model 512 \
        --batch_size 16 \
        --seq_len 96 \
        --pred_len $pred_len \
        --itr 1 \
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $weather_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_origin_seq $use_origin_seq \
        --use_linear $use_linear \
        --use_relu $use_relu \
      > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07102011_55/weather/'0_TaperFourierFormer_'ETTh1'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
done

fi