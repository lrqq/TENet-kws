import tensorflow as tf

# 路径到您的 .meta 文件和检查点文件
meta_file_path = 'save/tenet6/TENet6Model-26000.meta'  # .meta 文件路径
checkpoint_path = 'save/tenet6/TENet6Model-26000'  # 检查点文件的前缀路径

# 加载图结构
saver = tf.train.import_meta_graph(meta_file_path)

# 创建一个会话并恢复权重
with tf.Session() as sess:
    # 恢复权重
    saver.restore(sess, checkpoint_path)

    # 现在您可以进行推理、训练等操作
    # 比如，获取默认图
    graph = tf.compat.v1.get_default_graph()

    # 遍历图中的所有操作
    all_nodes = [n for n in tf.get_default_graph().as_graph_def().node]
    print(all_nodes)
    all_ops = tf.get_default_graph().get_operations()
    for op in sess.graph.get_operations():
    # 'op.type' tells you the type of operation
    # 'op.name' gives you the name of the operation
        if op.type == 'Placeholder':
            print("Input node name:", op.name)
    # 导出图为.pb文件
    # output_graph_def = tf.graph_util.convert_variables_to_constants(
    #     sess, graph.as_graph_def(), ["TENet6/squeeze_logit"])  # ["output_node_name"]是输出节点的名称

    # with tf.gfile.GFile("tenet6.pb", "wb") as f:  # 导出路径
    #     f.write(output_graph_def.SerializeToString())