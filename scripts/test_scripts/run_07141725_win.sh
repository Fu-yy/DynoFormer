source activate fuypy310t2cu118

# 绘制不同的图

use_x_mark_enc=0
front_use_decomp=0
use_space_merge=0


use_fourier=1
use_conv=1

use_origin_seq=1
use_linear=1
use_relu=1


seq_len=96
if [ ! -d "../../run_log" ]; then
    mkdir ../../run_log
fi
if [ ! -d "../../run_log/log_07141725_win" ]; then
    mkdir ../../run_log/log_07141725_win
fi
if [ ! -d "../../run_log/log_07141725_win/ETTm1" ]; then
    mkdir ../../run_log/log_07141725_win/ETTm1
fi
if [ ! -d "../../run_log/log_07141725_win/ETTh1" ]; then
    mkdir ../../run_log/log_07141725_win/ETTh1
fi
if [ ! -d "../../run_log/log_07141725_win/ETTm2" ]; then
    mkdir ../../run_log/log_07141725_win/ETTm2
fi

if [ ! -d "../../run_log/log_07141725_win/ETTh2" ]; then
    mkdir ../../run_log/log_07141725_win/ETTh2
fi
if [ ! -d "../../run_log/log_07141725_win/electricity" ]; then
    mkdir ../../run_log/log_07141725_win/electricity
fi

if [ ! -d "../../run_log/log_07141725_win/Exchange" ]; then
    mkdir ../../run_log/log_07141725_win/Exchange
fi

if [ ! -d "../../run_log/log_07141725_win/Solar" ]; then
    mkdir ../../run_log/log_07141725_win/Solar
fi

if [ ! -d "../../run_log/log_07141725_win/weather" ]; then
    mkdir ../../run_log/log_07141725_win/weather
fi

if [ ! -d "../../run_log/log_07141725_win/Traffic" ]; then
    mkdir ../../run_log/log_07141725_win/Traffic
fi

if [ ! -d "../../run_log/log_07141725_win/PEMS03" ]; then
    mkdir ../../run_log/log_07141725_win/PEMS03
fi

if [ ! -d "../../run_log/log_07141725_win/PEMS04" ]; then
    mkdir ../../run_log/log_07141725_win/PEMS04
fi

if [ ! -d "../../run_log/log_07141725_win/PEMS07" ]; then
    mkdir ../../run_log/log_07141725_win/PEMS07
fi
if [ ! -d "../../run_log/log_07141725_win/PEMS08" ]; then
    mkdir ../../run_log/log_07141725_win/PEMS08
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


  python -u  ../../run.py \
        --root_path exchange_rate \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id NewModelTimesNet'_Exchange_'96'_'$pred_len \
        --model NewModelTimesNet \
        --mode regre \
        --data Exchange \
        --features M \
        --freq d \
        --seq_len 96 \
        --pred_len 96 \
        --layer_nums 3 \
        --d_model 64 \
        --d_ff 128 \
        --itr 1 \
  > ../../run_log/log_07141725_win/Exchange/'0_NewModelTimesNet_'Exchange'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_'0.01.log 2>&1


if false;then
#  h1 2分解核  dmodel=dff=256

for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --model_id NewModelTimesNet'_ETTh1_'96'_'$pred_len \
        --root_path ETT-small \
        --mode regre \
        --data ETTh1 \
        --freq h \
        --features M \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 64 \
        --d_ff 64 \
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
      > ../../run_log/log_07141725_win/ETTh1/'0_NewModelTimesNet_'ETTh1'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1

  done


#  h2 2分解核  d_model= d_ff = 128

for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --root_path ETT-small \
        --model_id NewModelTimesNet'_ETTh2_'96'_'$pred_len \
        --mode regre \
        --data ETTh2 \
        --freq h \
        --features M \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 64 \
        --d_ff 64 \
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
      > ../../run_log/log_07141725_win/ETTh2/'0_NewModelTimesNet_'ETTh2'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1

  done




# m1 d_model= d_ff = 128

for pred_len in 96 192 336 720; do
  python -u  ../../run.py \
        --root_path ETT-small \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id NewModelTimesNet'_ETTm1_'96'_'$pred_len \
        --model NewModelTimesNet \
        --mode regre \
        --data ETTm1 \
        --features M \
        --freq t \
        --seq_len 96 \
        --pred_len 96 \
        --layer_nums 3 \
        --d_model 64 \
        --itr 1 \
  > ../../run_log/log_07141725_win/ETTm1/'0_NewModelTimesNet_'ETTm1'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_'0.01.log 2>&1

  done



#  m2 2分解核

# m2 d_model= d_ff = 128

for pred_len in 96 192 336 720; do
   python -u  ../../run.py \
        --root_path ETT-small \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id NewModelTimesNet'_ETTm2_'96'_'$pred_len \
        --model NewModelTimesNet \
        --mode regre \
        --data ETTm2 \
        --features M \
        --freq t \
        --seq_len 96 \
        --pred_len 96 \
        --layer_nums 3 \
        --d_model 64 \
        --itr 1 \
  > ../../run_log/log_07141725_win/ETTm2/'0_NewModelTimesNet_'ETTm2'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_'0.01.log 2>&1
  done



# ECL
for pred_len in 96 192 336 720 ; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id NewModelTimesNet'_ECL_'96'_'$pred_len \
        --model NewModelTimesNet \
        --root_path electricity \
        --mode regre \
        --data ECL \
        --features M \
        --freq t \
        --d_layers 1 \
        --e_layers 2 \
        --d_ff 64 \
        --d_model 32 \
        --batch_size 16 \
        --train_epochs 30 \
        --seq_len 96 \
        --patience 10 \
        --pred_len $pred_len \
        --itr 1 \
      > ../../run_log/log_07141725_win/electricity/'0_NewModelTimesNet_'electricity'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done

# Exchange d_model=d_ff=128

for pred_len in 96 192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id NewModelTimesNet'_Exchange_'96'_'$pred_len \
        --model NewModelTimesNet \
        --root_path exchange_rate \
        --mode regre \
        --data Exchange \
        --features M \
        --e_layers 2 \
        --freq d \
        --d_layers 1 \
        --d_model 64 \
        --d_ff 64 \
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
      > ../../run_log/log_07141725_win/Exchange/'0_NewModelTimesNet_'Exchange'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done




#  weather
for pred_len in 96 192 336 720 ; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id NewModelTimesNet'_WTH_'96'_'$pred_len \
        --model NewModelTimesNet \
        --root_path weather \
        --mode regre \
        --data WTH \
        --features M \
        --freq t \
        --d_layers 1 \
        --e_layers 2 \
        --d_ff 64 \
        --d_model 32 \
        --batch_size 16 \
        --train_epochs 30 \
        --seq_len 96 \
        --patience 10 \
        --pred_len $pred_len \
        --itr 1 \
      > ../../run_log/log_07141725_win/weather/'0_NewModelTimesNet_'weather'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done

fi