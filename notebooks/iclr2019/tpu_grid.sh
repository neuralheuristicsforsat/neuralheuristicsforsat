function train {
	DIFFICULTY="$1"
	TEMPLATE="$2"
	LEVEL_NUMBER="$3"
	ATTENTION="$4"
	TPU_NAME="ng-tpu-$5"
	MODEL_NAME="0315_series5a_sr${DIFFICULTY}_l${LEVEL_NUMBER}_a${ATTENTION}_t${TPU_NAME}"


	SESSION="$MODEL_NAME"
	tmux new-s -d -s "$SESSION"
	tmux send-keys -t "$SESSION:0" "cd deepsat/deepsat" Enter
	tmux send-keys -t "$SESSION:0" "source remote/setup.sh" Enter

	tmux send-keys -t "$SESSION:0" "MODEL_NAME=\"$MODEL_NAME\"" Enter
	tmux send-keys -t "$SESSION:0" "TPU_NAME=\"$TPU_NAME\"" Enter
	tmux send-keys -t "$SESSION:0" "ATTENTION=\"$ATTENTION\"" Enter
	tmux send-keys -t "$SESSION:0" "LEVEL_NUMBER=\"$LEVEL_NUMBER\"" Enter
	tmux send-keys -t "$SESSION:0" "$TEMPLATE" Enter

}

function sr30 {
	TEMPLATE='python neurosat_tpu.py --use_tpu=True --tpu=$TPU_NAME --train_file=gs://neural-guidance-tensorflow/sr30_75x1e4_uncom_train/train_2*.tfrecord --test_file=gs://neural-guidance-tensorflow/sr30_10x1e4_uncom_test/* --train_steps=1200000 --test_steps=80 --model_dir=gs://neural-guidance-tensorflow/$MODEL_NAME --export_dir=gs://neural-guidance-tensorflow/export/$MODEL_NAME --variable_number=30 --clause_number=300 --train_files_gzipped=False --batch_size=128 --export_model --attention=$ATTENTION --level_number=$LEVEL_NUMBER'
	train 30 "$TEMPLATE" "$1" "$2" "$3"
}

function sr50 {
	TEMPLATE='python neurosat_tpu.py --use_tpu=True --tpu=$TPU_NAME --train_file=gs://neural-guidance-tensorflow/sr_50_direct_from_prom/sr_50/train_1_sr_50.tfrecord.gz --test_file=gs://neural-guidance-tensorflow/sr_50_direct_from_prom/sr_50/train_2_sr_50.tfrecord.gz --train_steps=600000 --test_steps=1000 --model_dir=gs://neural-guidance-tensorflow/$MODEL_NAME --export_dir=gs://neural-guidance-tensorflow/export/$MODEL_NAME --variable_number=50 --clause_number=500 --train_files_gzipped=True --test_files_gzipped=True --batch_size=64 --export_model --attention=$ATTENTION --level_number=$LEVEL_NUMBER'
	train 50 "$TEMPLATE" "$1" "$2" "$3"
}

function sr70 {
	TEMPLATE='python neurosat_tpu.py --use_tpu=True --tpu=$TPU_NAME --train_file=gs://neural-guidance-tensorflow/sr_70_gz/train/train_1_sr_70.tfrecord.gz --test_file=gs://neural-guidance-tensorflow/sr_70_gz/test/test_9_sr_70.tfrecord.gz --train_steps=600000 --test_steps=1000 --model_dir=gs://neural-guidance-tensorflow/$MODEL_NAME --export_dir=gs://neural-guidance-tensorflow/export/$MODEL_NAME --variable_number=70 --clause_number=700 --train_files_gzipped=True --test_files_gzipped=True --batch_size=64 --attention=$ATTENTION --export_model --level_number=$LEVEL_NUMBER'
	train 70 "$TEMPLATE" "$1" "$2" "$3"
}

function sr100 {
	TEMPLATE='python neurosat_tpu.py --use_tpu=True --tpu=$TPU_NAME --train_file=gs://neural-guidance-tensorflow/sr_100_gz/train/train_1_sr_100.tfrecord.gz --test_file=gs://neural-guidance-tensorflow/sr_100_gz/test/test_9_sr_100.tfrecord.gz --train_steps=1200000 --test_steps=1000 --model_dir=gs://neural-guidance-tensorflow/$MODEL_NAME --export_dir=gs://neural-guidance-tensorflow/export/$MODEL_NAME --variable_number=100 --clause_number=1000 --train_files_gzipped=True --test_files_gzipped=True --batch_size=32 --attention=$ATTENTION --level_number=$LEVEL_NUMBER --export_model'
	train 100 "$TEMPLATE" "$1" "$2" "$3"
}

sr30 30 True 00
sr50 30 True 01
sr50 40 True 02
sr70 30 False 03
sr70 30 True 04
sr100 30 True 05
sr30 20 True 06
sr30 20 False 07
sr30 30 False 08
sr50 20 True 09
sr50 20 False 10
sr50 30 False 11
sr50 40 False 12
sr50 50 True 13
sr50 50 False 14
sr70 20 True 15 
sr70 20 False 16
sr70 40 False 17
sr70 40 True 18


