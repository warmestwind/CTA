import numpy as np
import tensorflow as tf


def dice_loss_w(pred, gt):
    eps = 1e-5
    loss = 0
    weights = [0.2, 0.4, 0.4]
    for i in range(3):
        intersection = tf.reduce_sum(pred[..., i] * gt[..., i])  # need same data type
        union = eps + tf.reduce_sum(pred[..., i]) + tf.reduce_sum(gt[..., i])
        loss += weights[i] * (-(2 * intersection / union) + 1)
    return loss

def tversky_loss(pred, gt, alpha = 0.5, beta= 0.5):
    eps = 1e-5
    loss = 0
    for i in range(3):
        intersection = tf.reduce_sum(pred[..., i] * gt[..., i])  # need same data type
                                                                # FP                                  #FN
        union = eps+ intersection + alpha * tf.reduce_sum(pred[..., i]* (1-gt[..., i])) + beta* tf.reduce_sum(gt[..., i] *(1-pred[..., i]))
        loss += intersection / union
    return 1 - loss / 3

def focal_tversky_loss(pred, gt, alpha = 0.5, beta= 0.5, gamma = 0.75):
    #alph: FP, beta: FN
    eps = 1e-5
    loss = 0
    for i in range(3):
        intersection = tf.reduce_sum(pred[..., i] * gt[..., i])  # need same data type
                                                                # FP                                  #FN
        union = eps+ intersection + alpha * tf.reduce_sum(pred[..., i]* (1-gt[..., i])) + beta* tf.reduce_sum(gt[..., i] *(1-pred[..., i]))
        loss += tf.pow((1 - intersection / union), gamma)
    return loss

def weighted_cross_entropy(pred, gt, n_classes=3):
    # flatten
    pred.set_shape((1, 256, 128, 128))
    gt.set_shape((1, 256, 128, 128, 3))

    # argmax no gradient, so gt shape not one-hot
    # target = tf.argmax(gt, axis=-1)

    target = tf.cast(tf.reshape(gt, [-1]), tf.int32)
    output = tf.reshape(pred, [-1, n_classes])  # [d*h*w, 3]

    # Calculate in-batch class counts and total counts
    target_one_hot = tf.one_hot(target, n_classes)  # [d*h*w, 3]
    counts = tf.cast(tf.reduce_sum(target_one_hot, axis=0), tf.float32) # [-1], [num_c0, num_c1, num_c2]
    total_counts = tf.reduce_sum(counts) # all pixel num

    # Compute balanced sample weights for every pixel, sum weights equals 1
    weights = (tf.ones_like(counts) * total_counts) / (counts * n_classes)

    # take every pixel weights from class weights
    weights = tf.gather(weights, target)

    return tf.losses.sparse_softmax_cross_entropy(target, output, weights)


# 3D voxel-wise softmax cross-entropy loss
def weighted_cross_entropy_nplike(pred, gt, n_classes=3):
    # gt_one_hot=tf.one_hot(gt, n_classes)
    loss = 0
    for i in range(n_classes):
        gti=gt[..., i]
        predi = pred[...,i]
        weight = tf.reduce_sum(gt) / tf.reduce_sum(gti) / n_classes
        loss = loss + -tf.reduce_mean(weight * gti * tf.log(tf.clip_by_value(predi, 0.005, 1)))
    return loss
