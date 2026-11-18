import tensorflow as tf

def pairwise_distance(feature, labels, num_classes, alpha_value, squared=False):
    """
    计算具有数值稳定性的成对距离矩阵。
    output[i, j] = || feature[i, :] - feature[j, :] ||_2 / alpha^(|| label[i] - label[j] || / num_class)
    Args:
      feature: 2-D Tensor of size `[number of data, feature dimension]`.
      squared: Boolean, whether or not to square the pairwise distances.
    Returns:
      pairwise_distances: 2-D Tensor of size `[number of data, number of data]`.
    """
    # 计算特征向量之间的成对平方距离
    pairwise_distances_squared = (
        tf.math.add(
            tf.math.reduce_sum(tf.math.square(feature), axis=[1], keepdims=True),
            tf.math.reduce_sum(tf.math.square(tf.transpose(feature)), axis=[0], keepdims=True),
        )
        - 2.0 * tf.matmul(feature, tf.transpose(feature))
    )

    # 修正数值稳定性（保证对角线为零，并避免取平方根时产生负值）
    pairwise_distances_squared = tf.math.maximum(pairwise_distances_squared, 0.0)

    # 计算标签之间的成对平方距离
    label_pairwise_distances_squared = (
        tf.math.add(
            tf.math.square(labels),
            tf.math.square(tf.transpose(labels))
        )
        - 2.0 * tf.matmul(labels, tf.transpose(labels))
    )

    # 根据标签距离调整特征距离
    pairwise_distances_squared = pairwise_distances_squared / tf.pow(alpha_value, tf.math.sqrt(
        label_pairwise_distances_squared) / num_classes)

    # 再次修正数值稳定性（保证对角线为零）
    pairwise_distances_squared = tf.math.maximum(pairwise_distances_squared, 0.0)

    # 根据是否平方返回距离
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = tf.math.sqrt(
            pairwise_distances_squared + tf.cast(tf.math.less_equal(pairwise_distances_squared, 0.0), dtype=tf.dtypes.float32) * 1e-16
        )

    # 设置对角线为零
    num_data = tf.shape(feature)[0]
    mask_offdiagonals = (tf.ones_like(pairwise_distances)
                         - tf.linalg.diag(tf.ones([num_data])))
    pairwise_distances = tf.math.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances


def lifted_struct_loss(alpha_value, margin, num_classes):
    """
    计算提升结构化损失。
    该损失促使小批量中的正距离（具有相同标签的一对嵌入之间）
    小于任何负距离（具有不同标签的一对嵌入之间），以相对于嵌入可微分的方式优化嵌入向量。
    See: https://arxiv.org/abs/1511.06452.
    Args:
      labels: 1-D tf.int32 `Tensor`.
      embeddings: 2-D float `Tensor` of embedding vectors.
      margin: Float, margin term in the loss definition.
    Returns:
      lifted_loss: tf.float32 scalar.
    """

    def loss(y_true, y_pred):
        labels = tf.convert_to_tensor(y_true, name="labels")
        embeddings = tf.convert_to_tensor(y_pred, name="embeddings")

        if embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16:
            precise_embeddings = tf.cast(embeddings, tf.dtypes.float32)
        else:
            precise_embeddings = embeddings

        # 将标签转换为 float32 并调整形状
        lshape = tf.shape(labels)
        labels = tf.cast(tf.reshape(labels, [lshape[0], 1]), tf.dtypes.float32)

        # 计算成对的平方距离矩阵
        pairwise_distances = pairwise_distance(precise_embeddings, labels, num_classes, alpha_value, squared=True)
        adjacency = tf.equal(labels, tf.transpose(labels))
        adjacency_not = tf.logical_not(adjacency)  # 反转以便我们只能选择负样本

        batch_size = tf.size(labels)

        # 计算 margin 与成对距离的差异
        diff = margin - pairwise_distances
        mask = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
        row_minimums = tf.reduce_min(diff, 1, keepdims=True)
        row_negative_maximums = tf.reduce_max(tf.multiply(diff - row_minimums, mask), 1, keepdims=True) + row_minimums

        max_elements = tf.maximum(row_negative_maximums, tf.transpose(row_negative_maximums))
        diff_tiled = tf.tile(diff, [batch_size, 1])
        mask_tiled = tf.tile(mask, [batch_size, 1])
        max_elements_vect = tf.reshape(tf.transpose(max_elements), [-1, 1])

        # 计算损失矩阵的一部分
        loss_exp_left = tf.reshape(
            tf.reduce_sum(tf.multiply(tf.exp(diff_tiled - max_elements_vect), mask_tiled), 1, keepdims=True),
            [batch_size, batch_size]
        )

        # 计算最终的损失矩阵
        loss_mat = max_elements + tf.math.log(loss_exp_left + tf.transpose(loss_exp_left) + 1e-16)
        loss_mat += pairwise_distances

        mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(tf.ones([batch_size]))

        num_positives = tf.reduce_sum(mask_positives) / 2.0

        # 计算最终的 Lifted Structured Loss
        lifted_loss = tf.truediv(0.25 * tf.reduce_sum(tf.maximum(tf.multiply(loss_mat, mask_positives), 0.0)), num_positives)

        return lifted_loss

    return loss