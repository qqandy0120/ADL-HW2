python3 multiple-choice/run_swag_infer.py \
  --overwrite_output \
  --do_eval \
  --max_seq_length 512 \
  --learning_rate 4e-5 \
  --output_dir ckpt\mc\infer \
  --train_file cache\swagformat_data\train.json
  --validation_file cache\swagformat_data\test.json \
  --tokenizer_name ckpt\mc\checkpoint-5000 \
  --model_name_or_path ckpt\mc\checkpoint-5000 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 1 \


python multiple-choice/run_swag_infer.py --overwrite_output --do_eval --max_seq_length 512 --learning_rate 4e-5 --output_dir ckpt\mc\infer --train_file cache\swagformat_data\train.json --validation_file cache\swagformat_data\test.json --tokenizer_name ckpt\mc\checkpoint-5000 --model_name_or_path ckpt\mc\checkpoint-5000 --per_device_eval_batch_size 2 --gradient_accumulation_steps 2 --num_train_epochs 1