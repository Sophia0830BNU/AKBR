dataset="MUTAG"
model="WL"
attn_type="channel"
log_file="${model}_${dataset}_${attn_type}.out"

nohup python3 -u main.py \
  --dataset $dataset \
  --lr 0.004 \
  --device 1 \
  --epochs 500 \
  --feature_hid 36 \
  --method $model \
  --iteration_num 2 \
  --attn_type $attn_type
  >> "$log_file" &
