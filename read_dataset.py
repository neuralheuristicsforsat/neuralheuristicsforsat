import itertools

import tensorflow as tf


def main():
    tf.enable_eager_execution()

    filename = "test.tfrecord"

    record_iterator = tf.python_io.tf_record_iterator(path=filename)

    for string_record in itertools.islice(record_iterator, 1):
        example = tf.train.Example()
        example.ParseFromString(string_record)

        # print(example)
        print(len(example.features.feature["inputs"].float_list.value))


if __name__ == '__main__':
    main()
