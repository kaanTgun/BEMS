import tensorflow as tf 
import keras 
from keras.layers import Conv2D, BatchNormalization, Activation, glorot_uniform, ZeroPadding2D, Add
import RCNN_utils as utils 
from keras.engine.topology import Layer



def identity_block(X, f, filters, stage, block):
    
    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def convolutional_block(X, f, filters, stage, block, s=2):
    
    #   The convolution block combines both the main path and the shortcut

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

###############################################################
#                           IOU
###############################################################
def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union

def intersection(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0: return 0
	return w*h

def iou(a, b):
	# a and b should be (x1,y1,x2,y2)

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0

	area_i = intersection(a, b)
	area_u = union(a, b, area_i)

	return float(area_i) / float(area_u + 1e-6)

class YOLOGrow(Layer):

    def __init__(self, anchors, max_grid, batch_size, warmup_batches, ignore_thresh, 
                    grid_scale, obj_scale, noobj_scale, xywh_scale, class_scale, 
                    **kwargs):
        # make the model settings persistent
        self.ignore_thresh  = ignore_thresh
        self.warmup_batches = warmup_batches
        self.anchors        = tf.constant(anchors, dtype='float', shape=[1,1,1,3,2])
        self.grid_scale     = grid_scale
        self.obj_scale      = obj_scale
        self.noobj_scale    = noobj_scale
        self.xywh_scale     = xywh_scale
        self.class_scale    = class_scale        

        # make a persistent mesh grid
        max_grid_h, max_grid_w = max_grid

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))
        self.cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [batch_size, 1, 1, 3, 1])

        super(YOLOGrow, self).__init__(**kwargs)

def classificationLoss():

def bbox_loss():

def mask_loss():
    
def ResNetCustom(
    nb_class, 
    anchors, 
    max_box_per_image, 
    max_grid, 
    batch_size, 
    warmup_batches,
    ignore_thresh,
    
    grid_scales,
    obj_scale,
    noobj_scale,
    xywh_scale,
    class_scale, 

    clsifer_loss,
    bbox_loss,
    mask_loss):

    
    input_image = keras.layers.Input(shape=(None, None, 3)) # net_h, net_w, 3
    true_boxes  = keras.layers.Input(shape=(1, 1, 1, max_box_per_image, 4))
    true_yolo_1 = keras.layers.Input(shape=(None, None, len(anchors)//6, 4+1+nb_class)) # grid_h, grid_w, nb_anchor, 5+nb_class
    true_yolo_2 = keras.layers.Input(shape=(None, None, len(anchors)//6, 4+1+nb_class)) # grid_h, grid_w, nb_anchor, 5+nb_class
    true_yolo_3 = keras.layers.Input(shape=(None, None, len(anchors)//6, 4+1+nb_class)) # grid_h, grid_w, nb_anchor, 5+nb_class


    # Stage 1
    X = Conv2D(32, kernel_size = (3, 3), strides = (1, 1), name = 'conv1_1', kernel_initializer = glorot_uniform(seed=0))(input_image)
    X = Activation('relu')(X)
    X = Conv2D(64, kernel_size = (3, 3), strides = (1, 1), name = 'conv1_2', kernel_initializer = glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters = [32, 64, 128], stage = 2, block='a', s = 1) # 224 x 224
    X = identity_block(X, 3, [64, 64, 128], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 128], stage=2, block='c')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='d')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='e')

    XStage2 = X

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage = 3, block='a', s = 2) # 112 x 112
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    XStage3 = X

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage = 4, block='a', s = 2) # 56 x 56
    X = identity_block(X, 3, [256, 512], stage=4, block='b')
    X = identity_block(X, 3, [256, 512], stage=4, block='c')
    X = identity_block(X, 3, [256, 512], stage=4, block='d')
    X = identity_block(X, 3, [256, 1024], stage=4, block='e')
    XStage4 = X

    # Stage 5
    X = convolutional_block(X, f=3, filters=[256, 512, 1024], stage = 5, block='a', s = 2) # 28 x 28
    X = identity_block(X, 3, [256, 512, 1024], stage=5, block='b')
    X = identity_block(X, 3, [256, 512, 1024], stage=5, block='c')
    X = identity_block(X, 3, [256, 512, 1024], stage=5, block='d')
    XStage5 = X


    # Stage 6 for upsampeling
    X = keras.layers.UpSampling2D(size=(2,2), name = 'DeConv6_1')(X)                        # 56 x 56
    XStage6 = keras.layers.Concatenate([XStage4, X])
    X = Conv2D(512, kernel_size = (1, 1), strides = (1, 1), name = 'conv6_1', kernel_initializer = glorot_uniform(seed=0))(XStage6)
    X = Conv2D(512, kernel_size = (2, 2), strides = (1, 1), name = 'conv6_2', kernel_initializer = glorot_uniform(seed=0))(X)
    X = Conv2D(512, kernel_size = (3, 3), strides = (1, 1), name = 'conv6_3', kernel_initializer = glorot_uniform(seed=0))(X)
    
    pred_yolo_1 = keras.layers.Conv2D(X,filter = (3*(5+nb_class)) , kernel =(1,1), bnorm = False, activation="bi")
    loss_yolo_1 = YOLOGrow(anchors[12:], 
                            [1*num for num in max_grid], 
                            batch_size, 
                            warmup_batches, 
                            ignore_thresh, 
                            grid_scales[0],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_1, true_yolo_1, true_boxes])



    # Stage 7 for upsampeling
    X = keras.layers.UpSampling2D(size=(2,2), name = 'DeConv7_1')(X)                       # 112 x 112
    XStage7 = keras.layers.Concatenate([XStage3, X])
    X = Conv2D(256, kernel_size = (1, 1), strides = (1, 1), name = 'conv7_1', kernel_initializer = glorot_uniform(seed=0))(XStage7)
    X = Conv2D(256, kernel_size = (3, 3), strides = (1, 1), name = 'conv7_2', kernel_initializer = glorot_uniform(seed=0))(X)

    pred_yolo_2 = keras.layers.Conv2D(X,filter = (3*(5+nb_class)) , kernel =(1,1), bnorm = False)
    loss_yolo_2 = YOLOGrow(anchors[6:12], 
                            [1*num for num in max_grid], 
                            batch_size, 
                            warmup_batches, 
                            ignore_thresh, 
                            grid_scales[1],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_2, true_yolo_2, true_boxes])

    # Stage 8 for upsampeling
    X = keras.layers.UpSampling2D(size=(2,2), name = 'DeConv8_1')(X)                        # 224 x 224
    XStage8 = keras.layers.Concatenate([XStage2, X])
    X = Conv2D(128, kernel_size = (1, 1), strides = (1, 1), name = 'conv8_1', kernel_initializer = glorot_uniform(seed=0))(XStage8)
    X = Conv2D(128, kernel_size = (3, 3), strides = (1, 1), name = 'conv8_2', kernel_initializer = glorot_uniform(seed=0))(X)
    X = Conv2D(128, kernel_size = (1, 1), strides = (1, 1), name = 'conv8_3', kernel_initializer = glorot_uniform(seed=0))(X)
    X = Conv2D(128, kernel_size = (3, 3), strides = (1, 1), name = 'conv8_4', kernel_initializer = glorot_uniform(seed=0))(X)

    pred_yolo_3 = keras.layers.Conv2D(X, filter = (3*(5+nb_class)), kernel =(1,1), bnorm = False)
    loss_yolo_3 = YOLOGrow(anchors[:6], 
                            [1*num for num in max_grid], 
                            batch_size, 
                            warmup_batches, 
                            ignore_thresh, 
                            grid_scales[2],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_3, true_yolo_3, true_boxes])


    train_model = keras.Model(  input = [input_image, true_boxes, true_yolo_1, true_yolo_2, true_yolo_3], 
                                output = [loss_yolo_1, loss_yolo_2, loss_yolo_3])

    infer_model = keras.Model(  input = input_image, 
                                output = [pred_yolo_1, pred_yolo_2, pred_yolo_3])

    return [train_model, infer_model]


def crop_and_resize(image, boxes, box_ind, crop_size, pad_border=True):
    """
    Aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.
    Args:
        image: NCHW
        boxes: nx4, x1y1x2y2
        box_ind: (n,)
        crop_size (int):
    Returns:
        n,C,size,size
    """
    assert isinstance(crop_size, int), crop_size
    boxes = tf.stop_gradient(boxes)

    # TF's crop_and_resize produces zeros on border
    if pad_border:
        # this can be quite slow
        image = tf.pad(image, [[0, 0], [0, 0], [1, 1], [1, 1]], mode='SYMMETRIC')
        boxes = boxes + 1

    @under_name_scope()
    def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
        """
        The way tf.image.crop_and_resize works (with normalized box):
        Initial point (the value of output[0]): x0_box * (W_img - 1)
        Spacing: w_box * (W_img - 1) / (W_crop - 1)
        Use the above grid to bilinear sample.
        However, what we want is (with fpcoor box):
        Spacing: w_box / W_crop
        Initial point: x0_box + spacing/2 - 0.5
        (-0.5 because bilinear sample (in my definition) assumes floating point coordinate
         (0.0, 0.0) is the same as pixel value (0, 0))
        This function transform fpcoor boxes to a format to be used by tf.image.crop_and_resize
        Returns:
            y1x1y2x2
        """
        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

        spacing_w = (x1 - x0) / tf.cast(crop_shape[1], tf.float32)
        spacing_h = (y1 - y0) / tf.cast(crop_shape[0], tf.float32)

        imshape = [tf.cast(image_shape[0] - 1, tf.float32), tf.cast(image_shape[1] - 1, tf.float32)]
        nx0 = (x0 + spacing_w / 2 - 0.5) / imshape[1]
        ny0 = (y0 + spacing_h / 2 - 0.5) / imshape[0]

        nw = spacing_w * tf.cast(crop_shape[1] - 1, tf.float32) / imshape[1]
        nh = spacing_h * tf.cast(crop_shape[0] - 1, tf.float32) / imshape[0]

        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

    image_shape = tf.shape(image)[2:]

    boxes = transform_fpcoor_for_tf(boxes, image_shape, [crop_size, crop_size])
    image = tf.transpose(image, [0, 2, 3, 1])   # nhwc
    ret = tf.image.crop_and_resize(
        image, boxes, tf.cast(box_ind, tf.int32),
        crop_size=[crop_size, crop_size])
    ret = tf.transpose(ret, [0, 3, 1, 2])   # ncss
    return ret

def roi_align(featuremap, boxes, resolution):
    """
    Args:
        featuremap: 1xCxHxW
        boxes: Nx4 floatbox
        resolution: output spatial resolution
    Returns:
        NxCx res x res
    """
    # sample 4 locations per roi bin
    ret = crop_and_resize(
        featuremap, boxes,
        tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32),
        resolution * 2)
    try:
        avgpool = tf.nn.avg_pool2d
    except AttributeError:
        avgpool = tf.nn.avg_pool
    ret = avgpool(ret, [1, 1, 2, 2], [1, 1, 2, 2], padding='SAME', data_format='NCHW')
    return ret


    # Output layer
    # X = Flatten()(X)
    # X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    # model = Model(inputs = X_input, outputs = X, name='ResNet50')

    # return model

if __name__ == "__main__":

    model, XStage1, XStage2, XStage3, XStage4 = ResNetCustom()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs = 2, batch_size = 32)

