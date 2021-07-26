import random
import tensorflow as tf
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))
import tf_util


def LSTM_Block(input_image, input_cell_state, hidden_features, is_training, bn_decay=None, scopeExt="0"):
    print('----------------------------hello world-----------------------------')
    input_feature = tf.concat([input_image, hidden_features], axis=-1)
    # input_state
    I = tf_util.conv2d(
        input_feature,
        32,
        [1, 1],
        padding="VALID",
        stride=[1, 1],
        bn=True,
        is_training=is_training,
        scope="conv1" + scopeExt,
        bn_decay=bn_decay,
        activation_fn=None
    )
    sigmoid_I = tf.nn.sigmoid(I)
    # forget_state
    F = tf_util.conv2d(
        input_feature,
        32,
        [1, 1],
        padding="VALID",
        stride=[1, 1],
        bn=True,
        is_training=is_training,
        scope="conv2" + scopeExt,
        bn_decay=bn_decay,
        activation_fn=None
    )
    sigmoid_F = tf.nn.sigmoid(F)
    candidate_cell = tf_util.conv2d(
        input_feature,
        32,
        [1, 1],
        padding="VALID",
        stride=[1, 1],
        bn=True,
        is_training=is_training,
        scope="conv3" + scopeExt,
        bn_decay=bn_decay,
        activation_fn=None
    )
    tanh_candidate_cell = tf.nn.tanh(candidate_cell)
    cell = sigmoid_F * input_cell_state + sigmoid_I * tanh_candidate_cell
    O = tf_util.conv2d(
        input_feature,
        32,
        [1, 1],
        padding="VALID",
        stride=[1, 1],
        bn=True,
        is_training=is_training,
        scope="conv4" + scopeExt,
        bn_decay=bn_decay,
        activation_fn=None
    )
    sigmoid_O = tf.nn.sigmoid(O)
    lstm_features = sigmoid_O * tf.nn.tanh(cell)
    attention_map = tf_util.conv2d(
        lstm_features,
        1,
        [1, 1],
        padding="VALID",
        stride=[1, 1],
        bn=True,
        is_training=is_training,
        scope="conv5" + scopeExt,
        bn_decay=bn_decay,
        activation_fn=None
    )
    attention_map = tf.nn.sigmoid(attention_map)
    ret = {
        'attention_map': attention_map,
        'cell': cell,
        'lstm_features': lstm_features
    }
    return ret


# select points by index
def select(input, index):
    print(index)
    len1 = input.get_shape()[0].value
    len2 = index.get_shape()[1].value
    last_index = input.get_shape()[3].value
    for i in range(len1):
        result = tf.gather(input[i], index[i], axis=0)
        if i == 0:
            output = result
        else:
            output = tf.concat([output, result], 0)
    output = tf.reshape(output, (len1, len2, -1, last_index))

    return output


# sample points by LA-Net
def get_points(
        point_cloud, is_training, num_output_points, bn_decay=None
):
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -2)
    temp_image = input_image
    # initial cell/hidden_features
    init_cell = tf.fill((batch_size, num_point, 1, 32), 0.0)
    hidden_features = tf.fill((batch_size, num_point, 1, 32), 0.0)
    # print('---init_cell---：',init_cell)
    for i in range(2):
        input_features = tf_util.conv2d(
            temp_image,
            32,
            [1, 1],
            padding="VALID",
            stride=[1, 1],
            bn=True,
            is_training=is_training,
            scope=f"conv_{i}",
            bn_decay=bn_decay,
        )
        lstm_ret = LSTM_Block(input_features, init_cell, hidden_features, is_training, bn_decay=None, scopeExt=f"{i}")
        attention_map = lstm_ret['attention_map']
        init_cell = lstm_ret['cell']
        hidden_features = lstm_ret['lstm_features']
        if i == 0:
            attention_key_value, attention_key_index = tf.nn.top_k(tf.squeeze(attention_map), num_point)
        else:
            attention_key_value, attention_key_index = tf.nn.top_k(tf.squeeze(attention_map), num_output_points)
        # Sample points
        attention_key_point = select(input_image, attention_key_index)
        init_cell = select(init_cell, attention_key_index)
        hidden_features = select(hidden_features, attention_key_index)
        temp_image = attention_key_point
    output_points = tf.squeeze(attention_key_point)
    return attention_key_index, output_points


# sample points by RS
def get_points1(point_cloud, is_training, num_output_points, bn_decay=None):
    len1 = point_cloud.get_shape()[0].value
    last_index = point_cloud.get_shape()[2].value
    for i in range(len1):
        index = random.sample(range(0, 1023), num_output_points)
        index = tf.reshape(index, [num_output_points])
        result = tf.gather(point_cloud[i], index, axis=0)
        if i == 0:
            tput_index = index
            output = result
        else:
            output_index = tf.concat([output_index, index], 0)
            output = tf.concat([output, result], 0)
    output_index = tf.reshape(output_index, (len1, num_output_points))
    output = tf.reshape(output, (len1, -1, last_index))
    return output_index, output


# sample points by FPS
def farthest_point_sample(point, npoint):
    N, D = point.shape
    centroids = tf.zeros(npoint)
    distance = tf.ones(N) * 1e10
    farthest = random.randint(0, 1023)
    farthest = tf.reshape(farthest, [1])
    for i in range(npoint):
        centroid = inputs[farthest]
        dist = tf.reduce_sum((inputs - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = tf.argmax(distance, -1)
    return centroids


if __name__ == '__main__':
    with tf.Session() as sess:
        inputs = tf.zeros((10000, 3))
        print(inputs[0])
