import tensorflow as tf
import matplotlib.pyplot as plt

image_raw_data = tf.gfile.FastGFile("C:\\Users\\xiech\\IdeaProjects\\AIDemo\\TF\\img.jpg","r").read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)

    print (img_data.eval())