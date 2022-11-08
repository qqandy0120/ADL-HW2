python question-answering/run_qa.py \
  --overwrite_output_dir \
  --model_name_or_path hfl/chinese-macbert-large \
  --tokenizer_name hfl/chinese-macbert-large \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --train_file cache\squadformat_data\train.json \
  --validation_file cache\squadformat_data\valid.json \
  --learning_rate 4e-5 \
  --num_train_epochs 1 \
  --max_seq_length 512 \
  --output_dir ckpt\qa





python question-answering/run_qa.py --overwrite_output_dir --model_name_or_path hfl/chinese-macbert-large --tokenizer_name hfl/chinese-macbert-large --do_train --do_eval --per_device_train_batch_size 2 --gradient_accumulation_steps 2 --train_file cache\squadformat_data\train.json --validation_file cache\squadformat_data\valid.json --learning_rate 4e-5 --num_train_epochs 2 --max_seq_length 512 --output_dir ckpt\qa