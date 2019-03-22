# Generating data

    dump_dataset.py -o <num_of_samples> -c <num_of_variables> -j <random_seed>
    
# Training

Run on TPU like this:

    python neurosat_tpu.py \
        --use_tpu=True \
        --tpu=$TPU_NAME \
        --train_file=$TRAINNIG_FILE \
        --test_file=$TEST_FILE \
        --train_steps=1200000 \
        --test_steps=80 \
        --model_dir=$MODEL_DIR \
        --export_dir=$EXPORT_DIR \
        --variable_number=30 \
        --clause_number=300 \   # 10 * variable_number
        --train_files_gzipped=False \
        --batch_size=128 \
        --export_model \
        --attention=$ATTENTION \
        --level_number=$LEVEL_NUMBER

Examples and hyperparameters can be read in `notebooks/iclr2019/tpu_grid.sh`.

# Evaluation with DPLL or CDCL

* For DPLL with 1000 step limit see `notebooks/iclr2019/dpll_1000_steps.ipynb`.
* For DPLL without a step limit see `notebooks/iclr2019/hybrid_dpll.ipynb`.
* For CDCL without a step limit see `notebooks/iclr2019/hybrid_cdcl.ipynb`.

