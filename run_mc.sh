python3 multiple-choice/run_swag.py \
  --overwrite_output \
  --do_train \
  --do_eval \
  --max_seq_length 512 \
  --learning_rate 3e-5 \
  --output_dir ckpt\mc \
  --train_file cache\swagformat_data\train.json \
  --validation_file cache\swagformat_data\train.json \
  --tokenizer_name hfl/chinese-macbert-base \
  --model_name_or_path hfl/chinese-macbert-base \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 1 \


# python multiple-choice/run_swag.py --overwrite_output --do_train --do_eval --max_seq_length 512 --learning_rate 4e-5 --output_dir ckpt\mc --train_file cache\swagformat_data\train.json --validation_file cache\swagformat_data\valid.json --tokenizer_name hfl/chinese-macbert-base --model_name_or_path hfl/chinese-macbert-base --per_device_train_batch_size 2 --gradient_accumulation_steps 2 --num_train_epochs 1