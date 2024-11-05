if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

>logs/etth1.log

gpu=0

for pred_len in 96 192 336 720
do
  model=MambaFormer
  label_len=192
  seq_len=$label_len
  python -u run_exp.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTh1.csv \
    --model_id ETTh1'_'$label_len'_'$pred_len \
    --model $model \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --embed_type 2 \
    --d_state 16 \
    --d_conv 4 \
    --d_layers 1 \
    --dec_in 7 \
    --c_out 7 \
    --batch_size 128 \
    --train_epochs 10 \
    --learning_rate 0.0001 \
    --lradj type1 \
    --des 'Exp' \
    --gpu $gpu \
    --itr 1 >>logs/etth1.log
done


for pred_len in 96 192 336 720
do
  model=AttMam
  label_len=192
  seq_len=$label_len
  python -u run_exp.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTh1.csv \
    --model_id ETTh1'_'$label_len'_'$pred_len \
    --model $model \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --embed_type 1 \
    --d_state 16 \
    --d_conv 4 \
    --d_layers 1 \
    --dec_in 7 \
    --c_out 7 \
    --batch_size 128 \
    --train_epochs 10 \
    --learning_rate 0.0001 \
    --lradj type1 \
    --des 'Exp' \
    --gpu $gpu \
    --itr 1 >>logs/etth1.log
done


for pred_len in 96 192 336 720
do
  model=MamAtt
  label_len=192
  seq_len=$label_len
  python -u run_exp.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTh1.csv \
    --model_id ETTh1'_'$label_len'_'$pred_len \
    --model $model \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --embed_type 2 \
    --d_state 16 \
    --d_conv 4 \
    --d_layers 1 \
    --dec_in 7 \
    --c_out 7 \
    --batch_size 128 \
    --train_epochs 10 \
    --learning_rate 0.0001 \
    --lradj type1 \
    --des 'Exp' \
    --gpu $gpu \
    --itr 1 >>logs/etth1.log
done
  

for pred_len in 96 192 336 720
do
  model=Mamba
  label_len=192
  seq_len=$label_len
  python -u run_exp.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTh1.csv \
    --model_id ETTh1'_'$label_len'_'$pred_len \
    --model $model \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --embed_type 2 \
    --d_state 16 \
    --d_conv 4 \
    --d_layers 1 \
    --dec_in 7 \
    --c_out 7 \
    --batch_size 128 \
    --train_epochs 10 \
    --learning_rate 0.0001 \
    --lradj type1 \
    --des 'Exp' \
    --gpu $gpu \
    --itr 1 >>logs/etth1.log
done


for pred_len in 96 192 336 720
do
  model=DecoderOnly
  label_len=192
  seq_len=$label_len
  python -u run_exp.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTh1.csv \
    --model_id ETTh1'_'$label_len'_'$pred_len \
    --model $model \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --embed_type 1 \
    --d_layers 1 \
    --dec_in 7 \
    --c_out 7 \
    --batch_size 128 \
    --train_epochs 10 \
    --learning_rate 0.0001 \
    --lradj type1 \
    --gpu $gpu \
    --des 'Exp' \
    --itr 1 >>logs/etth1.log
done

