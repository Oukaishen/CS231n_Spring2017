#MyModel record

tf.reset_default_graph()

def my_model(X,y,is_training):
    pass
    #kaishen's model
    #conv1  [N,32,32,3] -> [N,32,32,64]
    conv1 = tf.layers.conv2d(
    inputs = X,
    filters = 64,
    kernel_size=[3,3],
    padding="same",
    activation=None)
    
    #before the relu first BN
    #BN1
    bn1 = tf.layers.batch_normalization(inputs=conv1,training=is_training)

    relu1 = tf.nn.relu(bn1)
    
    conv2 = tf.layers.conv2d(
    inputs = relu1,
    filters = 64,
    kernel_size=[3,3],
    padding = "same",
    activation=None)
    
    bn2 = tf.layers.batch_normalization(inputs=conv2,training=is_training)
    
    relu2 = tf.nn.relu(bn2)

    #Max-pool, 2x2 stride 2   [N,16,16,32]
    pool1 = tf.layers.max_pooling2d(inputs=relu2,pool_size=[2,2],strides=2)
    
    #conv3  [N,16,16,32] -> [N,14,14,32]
    conv3 = tf.layers.conv2d(
    inputs = pool1,
    filters = 32,
    kernel_size=[3,3],
    padding="valid",
    activation=None)
    
    #BN3
    bn3 = tf.layers.batch_normalization(inputs=conv3,training=is_training)

    relu3 = tf.nn.relu(bn3)
    
    #Max-pool, 2x2 stride 2   [N,7,7,32]
    pool2 = tf.layers.max_pooling2d(inputs=relu3,pool_size=[2,2],strides=2,padding="same")
    
    pool2_flat = tf.reshape(pool2,[-1, 7*7*32])
    
    #dense
    dense = tf.layers.dense(inputs= pool2_flat,units=1024,activation=None)
    
    # first BN and then relu and then dropout
    bn4 = tf.layers.batch_normalization(inputs=dense,training=is_training)

    relu4 = tf.nn.relu(bn4)

    dropout= tf.layers.dropout(inputs=relu4,rate=0.4,training=is_training)

    #scores / logits 
    logits = tf.layers.dense(inputs=dropout, units=10)
    
    return logits