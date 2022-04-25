import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.applications as app

def conv_block(inp, ch_in, ch_out):
    x = layers.Conv2D(ch_in, 3, 1, 'same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(ch_out, 3, 1, 'same')(inp)
    x = layers.BatchNormalization()(x)
    out = layers.ReLU()(x)
    return out

def up_conv(inp, ch_in, ch_out):
    x = layers.UpSampling2D(2)(inp)
    x = layers.Conv2D(ch_out, 3, 1, 'same')(x)
    x = layers.BatchNormalization()(x)
    out = layers.ReLU()(x)
    return out

def attention_block(inp_x, inp_g, ch_out):
    x = layers.Conv2D(ch_out, 1, 1, 'same')(inp_x)
    x = layers.BatchNormalization()(x)
    g = layers.Conv2D(ch_out, 1, 1, 'same')(inp_g)
    g = layers.BatchNormalization()(g)
    f = layers.Add()([x, g])
    f = layers.ReLU()(f)
    f = layers.Conv2D(ch_out, 1, 1, 'same')(f)
    f = layers.BatchNormalization()(f)
    f = tf.nn.sigmoid(f)
    return f

class UNet:
    def __init__(self, input_size, base_model=None):
        self.input_size = input_size
        self.channels = [32, 64, 128, 256, 512]
        if base_model == 'mobilenet':
            self.base = app.MobileNetV2(input_shape=input_size, include_top=False, weights='imagenet')
            self.out_layers = [
                'block_1_expand_relu',   # 64x64
                'block_3_expand_relu',   # 32x32
                'block_6_expand_relu',   # 16x16
                'block_13_expand_relu',  # 8x8
                'block_16_project',      # 4x4
            ]
        elif base_model == 'efficientnet':
            self.base = app.EfficientNetB0(input_shape=input_size, include_top=False, weights='imagenet')
            self.out_layers = ['top_activation', 'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation']
        else:
            self.base = None
    
    def make_model(self):
        if self.base is None:
            inp = layers.Input(shape=self.input_size, dtype=tf.float32)
            # Encode path
            x1 = conv_block(inp, 3, self.channels[0]) #64x320

            x2 = layers.MaxPooling2D(2, 2)(x1)
            x2 = conv_block(x2, self.channels[0], self.channels[1]) #32x160

            x3 = layers.MaxPooling2D(2, 2)(x2)
            x3 = conv_block(x3, self.channels[1], self.channels[2]) #16x80

            x4 = layers.MaxPooling2D(2, 2)(x3)
            x4 = conv_block(x4, self.channels[2], self.channels[3]) #8x40

            x5 = layers.MaxPooling2D(2, 2)(x4)
            x5 = conv_block(x5, self.channels[3], self.channels[4]) #4x20
        else:
            inp = self.base.input
            x1, x2, x3, x4, x5 = [self.base.get_layer(l).output for l in self.out_layers]
                
        #decode path
        d5 = up_conv(x5, self.channels[4], self.channels[3])
#         d5 = layers.Add()([x4, d5])
        d5 = layers.Concatenate(axis=-1)([x4, d5])
        d5 = conv_block(d5, self.channels[4], self.channels[3])

        d4 = up_conv(d5, self.channels[3], self.channels[2])
#         d4 = layers.Add()([x3, d4])
        d4 = layers.Concatenate(axis=-1)([x3, d4])
        d4 = conv_block(d4, self.channels[3], self.channels[2])

        d3 = up_conv(d4, self.channels[2], self.channels[1])
#         d3 = layers.Add()([x2, d3])
        d3 = layers.Concatenate(axis=-1)([x2, d3])
        d3 = conv_block(d3, self.channels[2], self.channels[1])

        d2 = up_conv(d3, self.channels[1], self.channels[0])
#         d2 = layers.Add()([x1, d2])
        d2 = layers.Concatenate(axis=-1)([x1, d2])
        d2 = conv_block(d2, self.channels[1], self.channels[0])
        
        d1 = layers.Conv2DTranspose(1, 3, strides=2, padding='same')(d2)
#         d1 = layers.Conv2D(1, 1, 1, 'valid')(d2)
        d1 = tf.nn.sigmoid(d1)
#         out = tf.squeeze(d1, -1)

        return tf.keras.models.Model(inputs=inp, outputs=d1)

class Attn_Unet:
    def __init__(self, input_size, base_model=None):
        self.input_size = input_size
        self.channels = [32, 64, 128, 256, 512]
        if base_model == 'mobilenet':
            self.base = app.MobileNetV2(input_shape=input_size, include_top=False, weights='imagenet')
            self.out_layers = [
                'block_1_expand_relu',   # 64x64
                'block_3_expand_relu',   # 32x32
                'block_6_expand_relu',   # 16x16
                'block_13_expand_relu',  # 8x8
                'block_16_project',      # 4x4
            ]
        elif base_model == 'efficientnet':
            self.base = app.EfficientNetB0(input_shape=input_size, include_top=False, weights='imagenet')
            self.out_layers = ['top_activation', 'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation']
        else:
            self.base = None
    
    def make_model(self):
        if self.base is None:
            inp = layers.Input(shape=self.input_size, dtype=tf.float32)
            # Encode path
            x1 = conv_block(inp, 3, self.channels[0]) #64x320

            x2 = layers.MaxPooling2D(2, 2)(x1)
            x2 = conv_block(x2, self.channels[0], self.channels[1]) #32x160

            x3 = layers.MaxPooling2D(2, 2)(x2)
            x3 = conv_block(x3, self.channels[1], self.channels[2]) #16x80

            x4 = layers.MaxPooling2D(2, 2)(x3)
            x4 = conv_block(x4, self.channels[2], self.channels[3]) #8x40

            x5 = layers.MaxPooling2D(2, 2)(x4)
            x5 = conv_block(x5, self.channels[3], self.channels[4]) #4x20
        else:
            inp = self.base.input
            x1, x2, x3, x4, x5 = [self.base.get_layer(l).output for l in self.out_layers]
                
        #decode path
        d5 = up_conv(x5, self.channels[4], self.channels[3])
#         d5 = layers.Add()([x4, d5])
        x4 = attention_block(inp_x=x4, inp_g=d5, ch_out=int(d5.shape[-1]))
        d5 = layers.Concatenate(axis=-1)([x4, d5])
        d5 = conv_block(d5, self.channels[4], self.channels[3])

        d4 = up_conv(d5, self.channels[3], self.channels[2])
#         d4 = layers.Add()([x3, d4])
        x3 = attention_block(inp_x=x3, inp_g=d4, ch_out=int(d4.shape[-1]))
        d4 = layers.Concatenate(axis=-1)([x3, d4])
        d4 = conv_block(d4, self.channels[3], self.channels[2])

        d3 = up_conv(d4, self.channels[2], self.channels[1])
#         d3 = layers.Add()([x2, d3])
        x2 = attention_block(inp_x=x2, inp_g=d3, ch_out=int(d3.shape[-1]))
        d3 = layers.Concatenate(axis=-1)([x2, d3])
        d3 = conv_block(d3, self.channels[2], self.channels[1])

        d2 = up_conv(d3, self.channels[1], self.channels[0])
#         d2 = layers.Add()([x1, d2])
        x1 = attention_block(inp_x=x1, inp_g=d2, ch_out=int(d2.shape[-1]))
        d2 = layers.Concatenate(axis=-1)([x1, d2])
        d2 = conv_block(d2, self.channels[1], self.channels[0])
        
        d1 = layers.Conv2DTranspose(21, 3, strides=2, padding='same')(d2)
#         d1 = layers.Conv2D(1, 1, 1, 'valid')(d2)
        d1 = tf.nn.softmax(d1)
#         out = tf.squeeze(d1, -1)

        return tf.keras.models.Model(inputs=inp, outputs=d1)

class EffAttnUnet:
    def __init__(self, input_size, base_model=None):
        self.input_size = input_size
        self.channels = [64, 128, 256, 512]
        self.base = app.EfficientNetB0(input_shape=input_size, include_top=False, weights='imagenet')
        self.out_layers = ['block2a_expand_activation', 'block3a_expand_activation',
                           'block4a_expand_activation', 'block6a_expand_activation']
    
    def make_model(self):
        inp = self.base.input
        x1, x2, x3, x4 = [self.base.get_layer(l).output for l in self.out_layers]
        #decode path
        d4 = up_conv(x4, self.channels[3], self.channels[2])
        x3 = attention_block(inp_x=x3, inp_g=d4, ch_out=int(d4.shape[-1]))
        d4 = layers.Concatenate(axis=-1)([x3, d4])
        d4 = conv_block(d4, self.channels[3], self.channels[2])

        d3 = up_conv(d4, self.channels[2], self.channels[1])
        x2 = attention_block(inp_x=x2, inp_g=d3, ch_out=int(d3.shape[-1]))
        d3 = layers.Concatenate(axis=-1)([x2, d3])
        d3 = conv_block(d3, self.channels[2], self.channels[1])

        d2 = up_conv(d3, self.channels[1], self.channels[0])
        x1 = attention_block(inp_x=x1, inp_g=d2, ch_out=int(d2.shape[-1]))
        d2 = layers.Concatenate(axis=-1)([x1, d2])
        d2 = conv_block(d2, self.channels[1], self.channels[0])
        
        d1 = layers.Conv2DTranspose(21, 3, strides=2, padding='same')(d2)
#         d1 = layers.Conv2D(1, 1, 1, 'valid')(d2)
        d1 = tf.nn.softmax(d1)
#         out = tf.squeeze(d1, -1)

        return tf.keras.models.Model(inputs=inp, outputs=d1)

if __name__ == '__main__':
    import os
    import numpy as np
    from time import time
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#     base = app.EfficientNetB1(input_shape=(64, 320, 3), include_top=False, weights='imagenet')
#     base.summary()
#     unet = Attn_Unet((64, 320, 3), 'mobilenet').make_model()
    unet = EffAttnUnet((448, 448, 3)).make_model()
    unet.summary()
    data = np.random.random((5, 448, 448, 3))
   
    for d in data:
        s = time()
        res = unet(tf.expand_dims(d, 0))
        print(res.shape, time()-s)
    





