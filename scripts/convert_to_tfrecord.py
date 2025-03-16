import tensorflow as tf
import numpy as np
from preprocess import preprocess_data

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image, label):
    feature = {
        'image': _bytes_feature(image.tobytes()),
        'label': _int64_feature(label),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecord(images, labels, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for image, label in zip(images, labels):
            example = serialize_example(image, label)
            writer.write(example)

if __name__ == "__main__":
    dataset_folder = "data/raw/train"  # Update this path to your dataset location
    images, labels, label_encoder = preprocess_data(dataset_folder)
    write_tfrecord(images, labels, 'data/processed/dataset.tfrecord')



