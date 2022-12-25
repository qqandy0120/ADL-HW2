# transfer data into 2 formats
python preprocess.py --path_to_context "${1}" --path_to_test "${2}"

# inference mc
python ./multiple-choice/run_swag_infer.py \
  --overwrite_output \
  --do_eval \
  --max_seq_length 512 \
  --learning_rate 4e-5 \
  --output_dir ./ckpt/mc/infer \
  --train_file ./cache/swagformat_data/train.json \
  --validation_file ./cache/swagformat_data/test.json \
  --tokenizer_name ./ckpt/mc/checkpoint-5000 \
  --model_name_or_path ./ckpt/mc/checkpoint-5000 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 1 \
  --config_name ./ckpt/mc/checkpoint-5000 \
  --report_to none \

# inference qa
python ./question-answering/run_qa.py \
  --model_name_or_path ./ckpt/qa/checkpoint-13500 \
  --do_predict \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --train_file ./cache/squadformat_data/train.json \
  --validation_file ./cache/squadformat_data/valid.json \
  --test_file ./cache/squadformat_data/test.json \
  --learning_rate 4e-5 \
  --num_train_epochs 2 \
  --max_seq_length 512 \
  --output_dir ./ckpt/qa/infer \
  --config_name ./ckpt/qa/checkpoint-13500 \
  --tokenizer_name ./ckpt/qa/checkpoint-13500 \
  --report_to none \

python ./get_result.py --pred_path "${3}"