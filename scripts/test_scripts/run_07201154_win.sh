source activate fuypy310t2cu118

# 测试不同的参数

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
if [ ! -d "../../run_log/log_07201154_win" ]; then
    mkdir ../../run_log/log_07201154_win
fi
if [ ! -d "../../run_log/log_07201154_win/ETTm1" ]; then
    mkdir ../../run_log/log_07201154_win/ETTm1
fi
if [ ! -d "../../run_log/log_07201154_win/ETTh1" ]; then
    mkdir ../../run_log/log_07201154_win/ETTh1
fi
if [ ! -d "../../run_log/log_07201154_win/ETTm2" ]; then
    mkdir ../../run_log/log_07201154_win/ETTm2
fi

if [ ! -d "../../run_log/log_07201154_win/ETTh2" ]; then
    mkdir ../../run_log/log_07201154_win/ETTh2
fi
if [ ! -d "../../run_log/log_07201154_win/electricity" ]; then
    mkdir ../../run_log/log_07201154_win/electricity
fi

if [ ! -d "../../run_log/log_07201154_win/Exchange" ]; then
    mkdir ../../run_log/log_07201154_win/Exchange
fi

if [ ! -d "../../run_log/log_07201154_win/Solar" ]; then
    mkdir ../../run_log/log_07201154_win/Solar
fi

if [ ! -d "../../run_log/log_07201154_win/weather" ]; then
    mkdir ../../run_log/log_07201154_win/weather
fi

if [ ! -d "../../run_log/log_07201154_win/Traffic" ]; then
    mkdir ../../run_log/log_07201154_win/Traffic
fi

if [ ! -d "../../run_log/log_07201154_win/PEMS03" ]; then
    mkdir ../../run_log/log_07201154_win/PEMS03
fi

if [ ! -d "../../run_log/log_07201154_win/PEMS04" ]; then
    mkdir ../../run_log/log_07201154_win/PEMS04
fi

if [ ! -d "../../run_log/log_07201154_win/PEMS07" ]; then
    mkdir ../../run_log/log_07201154_win/PEMS07
fi
if [ ! -d "../../run_log/log_07201154_win/PEMS08" ]; then
    mkdir ../../run_log/log_07201154_win/PEMS08
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

layer_nums=2
d_model=128
d_ff=128
down_sampling_layers=3
use_origin_seq=1
k=5
learning_rate=0.0005
for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --model_id NewModelTimesNet'_Exchange_'96'_'$pred_len \
        --root_path exchange_rate \
        --mode regre \
        --data Exchange \
        --freq h \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        --learning_rate $learning_rate \
        > ../../run_log/log_07201154_win/Exchange/'test_timestep_0_NewModelTimesNet_'Exchange'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_'0.01.log 2>&1
  done

if false;then
layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=5
batch_size=16
for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --model_id NewModelTimesNet'_Traffic_'96'_'$pred_len \
        --root_path Traffic \
        --mode regre \
        --data Traffic \
        --freq t \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --batch_size $batch_size \
        --pred_len $pred_len \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        --time_step $time_step \
        > ../../run_log/log_07201154_win/Traffic/'test_timestep_0_NewModelTimesNet_'Traffic'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_'0.01.log 2>&1
  done




#  h1 2分解核  dmodel=dff=256
#  h1 2分解核  dmodel=dff=256
layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=5
time_step=60
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
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        --time_step $time_step \
        > ../../run_log/log_07201154_win/ETTh1/'test_timestep_0_NewModelTimesNet_'ETTh1'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_'0.01.log 2>&1
  done


#  h2 ok
layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=4
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
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        --time_step $time_step \
        > ../../run_log/log_07201154_win/ETTh2/'test_timestep_0_NewModelTimesNet_'ETTh2'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_'0.01.log 2>&1
  done







fi




if false;then
layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=5
train_epochs=30
patience=10
learning_rate=0.001
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
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        --train_epochs $train_epochs \
        --patience $patience \
        --learning_rate $learning_rate \
        > ../../run_log/log_07201154_win/ETTh1/'test0_NewModelTimesNet_'ETTh1'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_'0.01.log 2>&1
  done


#  h1 2分解核  dmodel=dff=256
layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=2
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
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        > ../../run_log/log_07201154_win/ETTh1/'0_NewModelTimesNet_'ETTh1'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_'0.01.log 2>&1
  done


#  h2 ok
layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=3
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
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        > ../../run_log/log_07201154_win/ETTh2/'0_NewModelTimesNet_'ETTh2'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_'0.01.log 2>&1
  done




# m1 d_model= d_ff = 128
#  m1 ok
layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=3
for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --root_path ETT-small \
        --model_id NewModelTimesNet'_ETTm1_'96'_'$pred_len \
        --mode regre \
        --data ETTm1 \
        --freq t \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        > ../../run_log/log_07201154_win/ETTm1/'0_NewModelTimesNet_'ETTm1'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_'0.01.log 2>&1
  done


#  m2 2分解核
# m2   ok

layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=2
for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --root_path ETT-small \
        --model_id NewModelTimesNet'_ETTm2_'96'_'$pred_len \
        --mode regre \
        --data ETTm2 \
        --freq t \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        > ../../run_log/log_07201154_win/ETTm2/'0_NewModelTimesNet_'ETTm2'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_'0.01.log 2>&1
  done


#ECL
layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=2
for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --root_path electricity \
        --model_id NewModelTimesNet'_ECL_'96'_'$pred_len \
        --mode regre \
        --data ECL \
        --freq t \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        > ../../run_log/log_07201154_win/electricity/'0_NewModelTimesNet_'electricity'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_'0.01.log 2>&1
  done




# Exchange d_model=d_ff=128  ok

layer_nums=2
d_model=128
d_ff=128
down_sampling_layers=3
use_origin_seq=1
k=5
learning_rate=0.0005

for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --root_path exchange_rate \
        --model_id NewModelTimesNet'_Exchange_'96'_'$pred_len \
        --mode regre \
        --data Exchange \
        --freq d \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        --learning_rate $learning_rate \
        > ../../run_log/log_07201154_win/Exchange/'0_NewModelTimesNet_'Exchange'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_'0.01.log 2>&1
  done


#weather

layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=2
for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --root_path weather \
        --model_id NewModelTimesNet'_WTH_'96'_'$pred_len \
        --mode regre \
        --data WTH \
        --freq d \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        > ../../run_log/log_07201154_win/weather/'0_NewModelTimesNet_'weather'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_'0.01.log 2>&1
  done

  fi