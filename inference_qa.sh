# python question-answering/run_qa.py \
#   --model_name_or_path ckpt\qa\checkpoint-13500 \
#   --do_predict \
#   --per_device_train_batch_size 2 \
#   --per_device_eval_batch_size 2 \
#   --gradient_accumulation_steps 2 \
#   --train_file cache\squadformat_data\train.json \
#   --validation_file cache\squadformat_data\valid.json \
#   --test_file cache\squadformat_data\test.json \
#   --learning_rate 4e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length 512 \
#   --output_dir ckpt\qa\infer \




python question-answering/run_qa.py --model_name_or_path ckpt\qa\checkpoint-13500 --do_predict --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 2 --train_file cache\squadformat_data\train.json --validation_file cache\squadformat_data\valid.json --test_file cache\squadformat_data\test.json --learning_rate 4e-5 --num_train_epochs 2 --max_seq_length 512 --output_dir ckpt\qa\infer