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
    activation=tf.nn.relu)
    
    #BN1
    bn1 = tf.layers.batch_normalization(inputs=conv1,training=is_training)
    
    conv2 = tf.layers.conv2d(
    inputs = bn1,
    filters = 64,
    kernel_size=[3,3],
    padding = "same",
    activation=tf.nn.relu)
    
    bn2 = tf.layers.batch_normalization(inputs=conv2,training=is_training)
    
    #Max-pool, 2x2 stride 2   [N,16,16,32]
    pool1 = tf.layers.max_pooling2d(inputs=bn2,pool_size=[2,2],strides=2)
    
    #conv3  [N,16,16,32] -> [N,14,14,32]
    conv3 = tf.layers.conv2d(
    inputs = pool1,
    filters = 32,
    kernel_size=[3,3],
    padding="valid",
    activation=tf.nn.relu)
    
    #BN1
    bn3 = tf.layers.batch_normalization(inputs=conv3,training=is_training)
    
    #Max-pool, 2x2 stride 2   [N,7,7,32]
    pool2 = tf.layers.max_pooling2d(inputs=bn3,pool_size=[2,2],strides=2,padding="same")
    
    pool2_flat = tf.reshape(pool2,[-1, 7*7*32])
    
    #dense
    dense = tf.layers.dense(inputs= pool2_flat,units=1024,activation=tf.nn.relu)
    
    #dropout
    dropout= tf.layers.dropout(inputs=dense,rate=0.3,training=is_training)
    
    #scores / logits 
    logits = tf.layers.dense(inputs=dropout, units=10)
    
    return logits