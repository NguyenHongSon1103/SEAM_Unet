import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import cv2
from argparse import ArgumentParser
import os
import sys
from model import UNet, Attn_Unet, EffAttnUnet
from dataset import Dataset
import logging
from hparams import hparams
from losses import Semantic_loss_functions

os.environ['CUDA_VISIBLE_DEVICES'] = hparams['gpu']
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Get data
train_gen = Dataset(mode='train')
val_gen = Dataset(mode='val')
print("TRAINING DATA: %d SAMPLES"%(len(train_gen)*hparams['batch_size']))
print("VALIDATION DATA: %d SAMPLES"%(len(val_gen)*hparams['batch_size']))

# define model
model = Attn_Unet(hparams['image_shape'], 'mobilenet').make_model()
# model = EffAttnUnet(hparams['image_shape']).make_model()
with open(os.path.join(hparams['model_dir'], 'model.json'), 'w') as f:
    f.write(model.to_json())
# model.summary()
# define loss, optimizer and callbacks
SL = Semantic_loss_functions()
def loss_fn(ytrue, ypred):
    print(ytrue.shape, ypred.shape)
#     n, h, w, c = ypred.shape
#     ytrue_focal = tf.keras.layers.Reshape((h*w, c))(ytrue)
#     ypred_focal = tf.keras.layers.Reshape((h*w, c))(ypred)
    bce_loss = tf.keras.losses.categorical_crossentropy(
        ytrue, ypred, from_logits=False)
#     f_loss = tfa.losses.SigmoidFocalCrossEntropy(from_logits = False,
#     alpha = 0.2, gamma = 1.5)(ytrue_focal, ypred_focal)
#     f_loss = SL.focal_loss(ytrue, ypred)
    d_loss = SL.dice_loss(ytrue, ypred) 
#     t_loss = SL.focal_tversky(ytrue, ypred)
    return d_loss + tf.reduce_mean(bce_loss)
#     return tf.reduce_mean(f_loss) + d_loss

learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    hparams['base_lr'], 3000, hparams['end_lr'], power=0.9)

optim = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn,)
ckpt = tf.keras.callbacks.ModelCheckpoint(os.path.join(hparams['model_dir'],
                                          'weights.{epoch:02d}_{val_loss:.3f}.h5'),
                                          mode='min', monitor='loss',
                                          save_best_only=True, save_weights_only=True)
miou_fn = tf.keras.metrics.MeanIoU(21)
def miou(ytrue, ypred):
    ytrue_logits = tf.argmax(ytrue, -1)
    ypred_logits = tf.argmax(ypred, -1)
    return miou_fn(ytrue_logits, ypred_logits)

tfboard = tf.keras.callbacks.TensorBoard(hparams['log_dir'])
# train:
model.compile(loss=loss_fn, optimizer=optim, metrics=[miou])
model.fit(train_gen, epochs=hparams['epoch'], batch_size=hparams['batch_size'],
        callbacks=[ckpt, tfboard], validation_data=val_gen)