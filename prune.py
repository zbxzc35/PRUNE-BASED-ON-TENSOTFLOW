import tensorflow as tf
import numpy as np
import sys
import config

def get_variable_list(meta_path, data_path):
    v_list = {}
    for v in tf.trainable_variables():
        v_list[v.name[:-2]] = v
    return v_list

def prune_tf(variable, percent):
    tmp = abs(sess.run(variable.name))
    tmp = np.reshape(tmp, np.size(tmp))
    tmp.sort()
    thre = tmp[int(len(tmp) * float(percent))]
    array = sess.run(variable.name)
    under_threshold = abs(array) < thre
    if np.size(array) != 1:
        array[under_threshold] = 0
    count = np.sum(under_threshold)
    return array, ~under_threshold, count

def prune_tf_sparse(weight_arr, name, thresh=0.005):
    #assert isinstance(weight_arr, np.ndarray)

    under_threshold = abs(weight_arr) < thresh
    if np.size(weight_arr) != 1:
        weight_arr[under_threshold] = 0
    sess.run(variables_list[name].assign(np.transpose(weight_arr)))
    values = weight_arr[weight_arr != 0]
    indices = np.transpose(np.nonzero(weight_arr))
    shape = list(weight_arr.shape)

    count = np.sum(under_threshold)
    print("Non-zero count (Sparse %s): %s" % (name, weight_arr.size - count))
    return [indices, values, shape]

def apply_prune_on_grads(grads_and_vars, dict_nzidx):
    # Mask gradients with pruned elements
    for key, nzidx in dict_nzidx.items():
        count = 0
        for grad, var in grads_and_vars:
            if var.name == key+":0":
                nzidx_obj = tf.cast(tf.constant(nzidx), tf.float32)
                grads_and_vars[count] = (tf.multiply(nzidx_obj, grad), var)
            count += 1
    return grads_and_vars

def gen_sparse_dict(v_list):
    sparse_w = {}
    sparse_w.update(v_list)
    for v_name in v_list:
        target_arr = np.transpose(v_list[v_name].eval())
        sparse_arr = prune_tf_sparse(target_arr, v_name)
        sparse_w[str(v_name+"_idx")] = tf.Variable(tf.constant(sparse_arr[0], dtype=tf.int32), name=str(v_name+"_idx"))
        sparse_w[v_name] = tf.Variable(tf.constant(sparse_arr[1], dtype=tf.float32), name=v_name)
        sparse_w[v_name+"_shape"] = tf.Variable(tf.constant(sparse_arr[2], dtype=tf.int32), name=v_name+"_shape")
    return sparse_w

meta_path = config.meta_path
data_path = config.data_path
percent = config.percent
nzidx = {}
x = config.feature + ":0"
y = config.label + ":0"
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # Prune weight
    saver = tf.train.import_meta_graph(meta_path)
    saver.restore(sess, data_path)
    graph = tf.get_default_graph()
    variables_list = get_variable_list(meta_path, data_path)
    for v_name in variables_list:
        arr, nzidx[v_name], count = prune_tf(variables_list[v_name], percent)
        sess.run(variables_list[v_name].assign(arr))

    # Retrain networks
    cross_entropy = tf.get_collection('cross_entropy')[0]
    trainer = tf.train.AdamOptimizer(1e-4, name='retrain_trainer')
    grads_and_vars = trainer.compute_gradients(cross_entropy)
    grads_and_vars = apply_prune_on_grads(grads_and_vars, nzidx)
    retrain_step = trainer.apply_gradients(grads_and_vars)
    accuracy = tf.get_collection('accuracy')[0]
    for var in tf.global_variables():
        if tf.is_variable_initialized(var).eval() == False:
            sess.run(tf.variables_initializer([var]))
    for step in range(config.retrain_step):
        image, label = config.get_data()
        sess.run(retrain_step, feed_dict={x: image, y: label, "keep_prob:0": 1})
        if (step + 1) % 100 == 0:
            retrain_accuracy = accuracy.eval(feed_dict={x: image, y: label, "keep_prob:0": 1})
            print("step %d, training accuracy %g" % (step + 1, retrain_accuracy))

    # Prune
    sparse_w = gen_sparse_dict(variables_list)
    image, label = config.get_data()
    retrain_accuracy = accuracy.eval(feed_dict={x: image, y: label, "keep_prob:0": 1})
    print("After pruning, training accuracy %g" % (retrain_accuracy))
    # Initialize new variables in a sparse form
    for var in tf.global_variables():
        if tf.is_variable_initialized(var).eval() == False:
            sess.run(tf.variables_initializer([var]))

    # Save model objects to serialized format
    final_saver = tf.train.Saver(sparse_w)
    final_saver.save(sess, data_path + "_retrain")


