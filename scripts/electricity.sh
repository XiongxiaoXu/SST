if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Forecasting" ]; then
    mkdir ./logs/Forecasting
fi

root_path_name=./dataset/electricity
data_path_name=electricity.csv
model_id_name=electricity
data_name=custom

gpu=5
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
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --embed_type 4 \
    --m_layers 1 \
    --d_state 16 \
    --d_conv 4 \
    --m_patch_len 48 \
    --m_stride 16 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 16 \
    --stride 8 \
    --local_ws 7 \
    --concat 1 \
    --train_epochs 10 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --lradj 'SST'\
    --patience 10 \
    --pct_start 0.2\
    --des 'Exp' \
    --gpu $gpu \
    --itr 1 >>logs/Forecasting/$model_name'_'$model_id_name'_'$seq_len'_'$label_len'_'$pred_len.log 
done
