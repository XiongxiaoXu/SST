if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

>logs/exchange_rate.log

gpu=0

for pred_len in 96 192 336 720
do
  model=MambaFormer
  label_len=192
  seq_len=$label_len
  python -u run_exp.py \
    --is_training 1 \
    --root_path ./dataset/exchange'_'rate \
    --data_path exchange'_'rate.csv \
    --model_id exchange'_'rate'_'$label_len'_'$pred_len \
    --model $model \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --embed_type 2 \
    --d_state 16 \
    --d_conv 4 \
    --d_layers 1 \
    --dec_in 8 \
    --c_out 8 \
    --batch_size 64 \
    --train_epochs 10 \
    --learning_rate 0.001 \
    --lradj type1 \
    --des 'Exp' \
    --gpu $gpu \
    --itr 1 >>logs/exchange'_'rate.log
done


for pred_len in 96 192 336 720
do
  model=AttMam
  label_len=192
  seq_len=$label_len
  python -u run_exp.py \
    --is_training 1 \
    --root_path ./dataset/exchange'_'rate \
    --data_path exchange'_'rate.csv \
    --model_id exchange'_'rate'_'$label_len'_'$pred_len \
    --model $model \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --embed_type 1 \
    --d_state 16 \
    --d_conv 4 \
    --d_layers 1 \
    --dec_in 8 \
    --c_out 8 \
    --batch_size 64 \
    --train_epochs 10 \
    --learning_rate 0.001 \
    --lradj type1 \
    --des 'Exp' \
    --gpu $gpu \
    --itr 1 >>logs/exchange'_'rate.log
done


for pred_len in 96 192 336 720
do
  model=MamAtt
  label_len=192
  seq_len=$label_len
  python -u run_exp.py \
    --is_training 1 \
    --root_path ./dataset/exchange'_'rate \
    --data_path exchange'_'rate.csv \
    --model_id exchange'_'rate'_'$label_len'_'$pred_len \
    --model $model \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --embed_type 2 \
    --d_state 16 \
    --d_conv 4 \
    --d_layers 1 \
    --dec_in 8 \
    --c_out 8 \
    --batch_size 64 \
    --train_epochs 10 \
    --learning_rate 0.001 \
    --lradj type1 \
    --des 'Exp' \
    --gpu $gpu \
    --itr 1 >>logs/exchange'_'rate.log
done


for pred_len in 96 192 336 720
do
  model=Mamba
  label_len=192
  seq_len=$label_len
  python -u run_exp.py \
    --is_training 1 \
    --root_path ./dataset/exchange'_'rate \
    --data_path exchange'_'rate.csv \
    --model_id exchange'_'rate'_'$label_len'_'$pred_len \
    --model $model \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --embed_type 2 \
    --d_state 16 \
    --d_conv 4 \
    --d_layers 1 \
    --dec_in 8 \
    --c_out 8 \
    --batch_size 64 \
    --train_epochs 10 \
    --learning_rate 0.001 \
    --lradj type1 \
    --des 'Exp' \
    --gpu $gpu \
    --itr 1 >>logs/exchange'_'rate.log
done


for pred_len in 96 192 336 720
do
  model=DecoderOnly
  label_len=192
  seq_len=$label_len
  python -u run_exp.py \
    --is_training 1 \
    --root_path ./dataset/exchange'_'rate \
    --data_path exchange'_'rate.csv \
    --model_id exchange'_'rate'_'$label_len'_'$pred_len \
    --model $model \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --embed_type 1 \
    --d_layers 1 \
    --dec_in 8 \
    --c_out 8 \
    --batch_size 64 \
    --train_epochs 10 \
    --learning_rate 0.001 \
    --lradj type1 \
    --des 'Exp' \
    --gpu $gpu \
    --itr 1 >>logs/exchange'_'rate.log
done
