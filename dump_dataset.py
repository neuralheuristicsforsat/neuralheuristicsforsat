import tensorflow as tf
import numpy as np

import cnf_dataset
from tqdm import tqdm
import argparse
import os
import random


def make_example(inputs, sat, policy):
    example = tf.train.Example(
        features=tf.train.Features(feature={
            "inputs": tf.train.Feature(float_list=tf.train.FloatList(value=list(inputs.flatten()))),
            "sat": tf.train.Feature(
                float_list=tf.train.FloatList(value=list(sat.flatten()))),
            "policy": tf.train.Feature(
                float_list=tf.train.FloatList(value=list(policy.flatten())))
        })
    )
    return example.SerializeToString()


def tf_serialize_example(sample):
    tf_string = tf.py_func(make_example, (sample["inputs"], sample["sat"], sample["policy"]), tf.string)
    return tf.reshape(tf_string, ())


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--job", required=True, help="the job identifier")
    ap.add_argument("-c", "--complexity", required=True, type=int, default=30, help="the level of complexity of SR(n)")
    ap.add_argument("-o", "--observations", required=True, type=int, default=1e4, help="the number of observations to be made")
    args = vars(ap.parse_args())

    job = args["job"]
    random.seed(int(job))
    print("Set random seed to {}".format(int(job)))

    complexity = args["complexity"]
    dirname = "sr_{}".format(complexity)  
    filename = "train_{}_sr_{}.tfrecord".format(job, complexity)
    options = {
        "PROCESSOR_NUM": 24,
        "CLAUSE_NUM": 10*complexity,
        "VARIABLE_NUM": complexity,
        "MIN_VARIABLE_NUM": 1,
        "BATCH_SIZE": 1,
        "SR_GENERATOR": True
    }
    n_observations = args["observations"]

    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print("Created directory {}".format(dirname))

    with cnf_dataset.PoolDatasetGenerator(options) as generator, \
            tf.python_io.TFRecordWriter(os.path.join(dirname,filename)) as writer:

        for _ in tqdm(range(n_observations)):
            sample_with_labels = generator.generate_batch()
            tf_sample = {
                 "inputs": np.squeeze(sample_with_labels.inputs.astype(np.float32), 0),
                 "sat": np.squeeze(np.asarray(sample_with_labels.sat_labels).astype(np.float32), 0),
                 "policy": np.squeeze(sample_with_labels.policy_labels.astype(np.float32), 0)}

            serialized = make_example(**tf_sample)
            writer.write(serialized)


if __name__ == '__main__':
    main()
