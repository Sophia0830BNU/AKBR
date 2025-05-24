dataset="MUTAG"
attn_type="channel"
log_file="Sample_${dataset}_${attn_type}.out"

nohup python3 -u main_sample.py \
  --dataset $dataset \
  --lr 0.004 \
  --device 1 \
  --epochs 500 \
  --feature_hid 36 \
  --iteration_num 2 \
  --sample_number 200 \
  --attn_type $attn_type \
  >> "$log_file" &
