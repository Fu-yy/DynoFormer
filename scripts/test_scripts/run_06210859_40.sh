#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1
#SBATCH -w aiwkr1


#module load cuda/11.7.0
#module load singularity/3.11.0
module load cuda/11.8.0
module load anaconda/anaconda3-2022.10

source activate py310t2cu118



# 新   所有网络的参数





seq_len=96



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


use_x_mark_enc=0
front_use_decomp=0
use_space_merge=0


use_fourier=1
use_conv=1



if [ ! -d "/mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log" ]; then
    mkdir /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log
fi

if [ ! -d "/mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params" ]; then
    mkdir /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params
fi

if [ ! -d "/mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40" ]; then
    mkdir /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40
fi


# TaperFourierFormer ETTh1
for pred_len in 96  192 336 720; do
 python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model TaperFourierFormer \
      --root_path ETT-small \
      --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
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
      --down_sampling_layers 3 \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --channel_independence 1 \
      --use_space_merge $use_space_merge \
      --use_fourier $use_fourier \
      --front_use_decomp $front_use_decomp \
      --use_conv $use_conv \
      --use_x_mark_enc $use_x_mark_enc \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTh1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
#TimesNet ETTh1
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model TimesNet \
  --data_path ETTh1.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTh1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
#iTransformer ETTh1

for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --is_training 1 \
  --data_path ETTh1.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model iTransformer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTh1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done


# AutoFormer ETTh1

for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path ETTh1.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model AutoFormer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTh1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
#PatchTST ETTh1
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model PatchTST \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 2 \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTh1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
# Crossformer ETTh1
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path ETTh1.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model Crossformer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTh1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done


# TaperFourierFormer ETTh2
for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
        --root_path ETT-small \
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
    > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTh2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
#TimesNet ETTh2
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path ETTh2.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model TimesNet \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTh2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
#iTransformer ETTh2+

for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --is_training 1 \
  --data_path ETTh2.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model iTransformer \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTh2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done

# AutoFormer ETTh2
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path ETTh2.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model AutoFormer \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTh2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
#PatchTST ETTh2
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path ETTh2.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model PatchTST \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 4 \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTh2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
# Crossformer ETTh2
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path ETTh2.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model Crossformer \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTh2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done


# TaperFourierFormer ETTm1
  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
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
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
    > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTm1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
#TimesNet ETTm1
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path ETTm1.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model TimesNet \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 64 \
  --d_ff 64 \
  --top_k 5 \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTm1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done

#iTransformer ETTm1

for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --is_training 1 \
  --data_path ETTm1.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model iTransformer \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTm1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done




# AutoFormer ETTm1
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path ETTm1.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model AutoFormer \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTm1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
#PatchTST ETTm1
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path ETTm1.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model PatchTST \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 2 \
  --batch_size 32 \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTm1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
# Crossformer ETTm1
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path ETTm1.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model Crossformer \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTm1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done

# TaperFourierFormer ETTm2

  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run.py \
        --root_path ETT-small \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
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
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
#TimesNet ETTm2
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path ETTm2.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model TimesNet \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
#iTransformer ETTm2
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --is_training 1 \
  --data_path ETTm2.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model iTransformer \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
# AutoFormer ETTm2
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path ETTm2.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model AutoFormer \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
#PatchTST ETTm2
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path ETTm2.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model PatchTST \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 16 \
  --batch_size 32 \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
# Crossformer ETTm2
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path ETTm2.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model Crossformer \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done


# TaperFourierFormer ECL
 for pred_len in 96 192 336 720 ; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --root_path electricity \
        --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
        --mode regre \
        --data ECL \
        --freq h \
        --features M \
        --e_layers 3 \
        --d_layers 1 \
        --d_model 512 \
        --d_ff 512 \
        --batch_size 16 \
        --seq_len 96 \
        --pred_len $pred_len \
        --itr 1 \
        --learning_rate 0.0005 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $electricity_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'electricity'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
#TimesNet ECL
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --data_path electricity.csv \
    --model TimesNet \
    --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --d_model 256 \
    --d_ff 512 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'electricity'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
#iTransformer ECL
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path electricity.csv \
  --model iTransformer \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'electricity'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
# AutoFormer ECL
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path electricity.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model Autoformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'electricity'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
#PatchTST ECL
for pred_len in 96  192 336 720; do
python -  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path electricity.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model PatchTST \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 16 \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'electricity'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
# Crossformer ECL
for pred_len in 96  192 336 720; do
  python -  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path electricity.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model Crossformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --batch_size 16 \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'electricity'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done



# TaperFourierFormer Exchange
for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --root_path exchange_rate \
        --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
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
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $Exchange_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'Exchange'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
#TimesNet Exchange

for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
   --task_name long_term_forecast \
  --is_training 1 \
  --data_path exchange_rate.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model TimesNet \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 64 \
  --d_ff 64 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'Exchange'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done

#iTransformer Exchange
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --is_training 1 \
  --data_path exchange_rate.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'Exchange'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done

# AutoFormer Exchange
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path exchange_rate.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model AutoFormer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'Exchange'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
#PatchTST Exchange
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path exchange_rate.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model PatchTST \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'Exchange'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
# Crossformer Exchange
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path exchange_rate.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model Crossformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 96 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 64 \
  --d_ff 64 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'Exchange'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done


# TaperFourierFormer Weather
for pred_len in 96 192 336 720 ; do
   python -u  /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --root_path weather \
        --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
        --mode regre \
        --data WTH \
        --features M \
        --freq t \
        --d_layers 1 \
        --e_layers 3 \
        --d_ff 512 \
        --d_model 512 \
        --seq_len 96 \
        --pred_len $pred_len \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $weather_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'weather'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
#TimesNet Weather
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path weather.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model TimesNet \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'weather'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
#iTransformer Weather

for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --is_training 1 \
  --data_path weather.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'weather'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
# AutoFormer Weather
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path weather.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model AutoFormer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 2 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'weather'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
#PatchTST Weather
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path weather.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model PatchTST \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --n_heads 4 \
  --train_epochs 3 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'weather'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done
# Crossformer Weather
for pred_len in 96  192 336 720; do
  python -u /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_count_params.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path weather.csv \
  --model_id $model'_'$task_name'_'$seq_len'_'$pred_len \
  --model Crossformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  > /mnt/nfs/data/home/1120231440/home/fuy/python/time_series_40/run_log/params/log_06210859_40/'0'_$model_name'_'weather'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done






