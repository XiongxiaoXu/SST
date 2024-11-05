if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Forecasting" ]; then
    mkdir ./logs/Forecasting
fi

root_path_name=./dataset/weather
data_path_name=weather.csv
model_id_name=weather
data_name=custom

gpu=4
random_seed=2021
model_name=SST
label_len=336
seq_len=$(($label_len*2))

for pred_len in 96 192 336 720
do
  python -u run.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$label_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --embed_type 4 \
    --m_layers 1 \
    --d_state 16 \
    --d_conv 4 \
    --m_patch_len 48 \
    --m_stride 16 \
    --e_layers 3 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 128 \
    --dropout 0.3 \
    --fc_dropout 0.3 \
    --head_dropout 0 \
    --patch_len 16 \
    --stride 8 \
    --local_ws 7 \
    --concat 1 \
    --train_epochs 10 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --lradj type3 \
    --patience 10 \
    --des 'Exp' \
    --gpu $gpu \
    --itr 1 >>logs/Forecasting/$model_name'_'$model_id_name'_'$seq_len'_'$label_len'_'$pred_len.log 
done
