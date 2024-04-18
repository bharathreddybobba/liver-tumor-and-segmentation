import json
import math
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models as sm
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as tfl
import tensorflow_addons as tfa
import wandb
from kaggle_secrets import UserSecretsClient
from wandb.keras import WandbMetricsLogger

# Configuration parameters
CONF = {
    'a_min': -110, # clip min
    'a_max': 200,
    'batch_size': 64,
    'lr': 1e-3,
    'epochs': 500,
    'img_size': [256, 256],
    'channels': 5,
    'fill_mode': 'constant',
    'rot': 5.0,  # proprtional
    'shr': 5.0,  # proprtional
    'hzoom': 100.0,  # inv proportional
    'wzoom': 100.0,  # inv proportional
    'hshift': 10.0,  # proportional
    'wshift': 10.0,  # proportional
    'hflip': 0.5,
    'vflip': 0.5,
    'drop_prob': 0.5,
    'drop_cnt': 10,
    'drop_size': 0.05,
    'sat': [0.7, 1.3],  # saturation
    'cont': [0.8, 1.2],  # contrast
    'bri': 0.15,  # brightness
    'hue': 0.0,
    'wandb_on': True,
    'architecture': 'Unet 5s',
    'loss': 'DiceLoss',
    'metric': 'FScore',
    'lr_scheduler': 'CosineDecay',
    'augment': True
}

# Fetch secret keys
user_secrets = UserSecretsClient()
WANDB_KEY = user_secrets.get_secret('wandb')

# List files on GCS
GCS_PATH = user_secrets.get_secret('couinaud-segmentation-data')
DIR_PATH = f"{GCS_PATH}/r{CONF['img_size'][0]}-{CONF['img_size'][1]}-{CONF['channels']}"
TRAIN_SET = tf.io.gfile.glob(f'{DIR_PATH}/train_set/*.tfrecords')
DEV_SET = tf.io.gfile.glob(f'{DIR_PATH}/dev_set/*.tfrecords')
TEST_SET = tf.io.gfile.glob(f'{DIR_PATH}/test_set/*.tfrecords')

# Get size of sets
with tf.io.gfile.GFile(f'{DIR_PATH}/size.json', 'r') as f:
    SET_SIZE = json.load(f)

AUTO = tf.data.experimental.AUTOTUNE

# Configure device
def configure_device():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        strategy = tf.distribute.TPUStrategy(tpu)
        print('> Running on TPU ', tpu.master(), end=' | ')
        print('Num of TPUs: ', strategy.num_replicas_in_sync)
        device='TPU'
    except:
        tpu = None
        gpus = tf.config.list_logical_devices('GPU')
        ngpu = len(gpus)
        if ngpu:
            strategy = tf.distribute.MirroredStrategy(gpus)
            print("> Running on GPU", end=' | ')
            print("Num of GPUs: ", ngpu)
            device='GPU'
        else:
            print("> Running on CPU")
            strategy = tf.distribute.get_strategy()
            device='CPU'
    return strategy, device, tpu

strategy, device, tpu = configure_device()
AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')

# Utility functions
def random_int(shape=[], minval=0, maxval=1):
    return tf.random.uniform(shape=shape, minval=minval, maxval=maxval, dtype=tf.int32)

def random_float(shape=[], minval=0.0, maxval=1.0):
    rnd = tf.random.uniform(shape=shape, minval=minval, maxval=maxval, dtype=tf.float32)
    return rnd

def get_mat(shear, height_zoom, width_zoom, height_shift, width_shift):
    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst], axis=0), [3, 3])
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = get_3x3_mat([one, s2, zero, zero, c2, zero, zero, zero, one])
    zoom_matrix = get_3x3_mat([one / height_zoom, zero, zero, zero, one / width_zoom, zero, zero, zero, one])
    shift_matrix = get_3x3_mat([one, zero, height_shift, zero, one, width_shift, zero, zero, one])
    return K.dot(shear_matrix, K.dot(zoom_matrix, shift_matrix))

def ShiftScaleRotate(image, mask=None, DIM=CONF['img_size'], p=1.0):
    if random_float() > p:
        return image, mask
    if DIM[0] > DIM[1]:
        diff = DIM[0] - DIM[1]
        pad = [diff // 2, diff // 2 + (1 if diff % 2 else 0)]
        image = tf.pad(image, [[0, 0], [pad[0], pad[1]], [0, 0]])
        NEW_DIM = DIM[0]
        if mask is not None:
            mask = tf.pad(mask, [[0, 0], [pad[0], pad[1]], [0, 0]])
    elif DIM[0] < DIM[1]:
        diff = DIM[1] - DIM[0]
        pad = [diff // 2, diff // 2 + (1 if diff % 2 else 0)]
        image = tf.pad(image, [[pad[0], pad[1]], [0, 0], [0, 0]])
        NEW_DIM = DIM[1]
        if mask is not None:
            mask = tf.pad(mask, [[pad[0], pad[1]], [0, 0], [0, 0]])
    rot = CONF['rot'] * tf.random.normal([1], dtype="float32")
    shr = CONF['shr'] * tf.random.normal([1], dtype="float32")
    h_zoom = 1.0 + tf.random.normal([1], dtype="float32") / CONF['hzoom']
    w_zoom = 1.0 + tf.random.normal([1], dtype="float32") / CONF['wzoom']
    h_shift = CONF['hshift'] * tf.random.normal([1], dtype="float32")
    w_shift = CONF['wshift'] * tf.random.normal([1], dtype="float32")
    transformation_matrix = tf.linalg.inv(get_mat(shr, h_zoom, w_zoom, h_shift, w_shift))
    flat_tensor = tfa.image.transform_ops.matrices_to_flat_transforms(transformation_matrix)
    rotation = math.pi * rot / 180.0
    image = tfa.image.transform(image, flat_tensor, fill_mode=CONF['fill_mode'])
    image = tfa.image.rotate(image, -rotation, fill_mode=CONF['fill_mode'])
    if mask is not None:
        mask = tfa.image.transform(mask, flat_tensor, fill_mode=CONF['fill_mode'])
        mask = tfa.image.rotate(mask, -rotation, fill_mode=CONF['fill_mode'])
    if DIM[0] > DIM[1]:
        image = tf.reshape(image, [NEW_DIM, NEW_DIM, CONF["channels"]])
        image = image[:, pad[0] : -pad[1], :]
        if mask is not None:
            mask = tf.reshape(mask, [NEW_DIM, NEW_DIM, 9])
            mask = mask[:, pad[0] : -pad[1], :]
    elif DIM[1] > DIM[0]:
        image = tf.reshape(image, [NEW_DIM, NEW_DIM, CONF["channels"]])
        image = image[pad[0] : -pad[1], :, :]
        if mask is not None:
            mask = tf.reshape(mask, [NEW_DIM, NEW_DIM, 9])
            mask = mask[pad[0] : -pad[1], :, :]
    image = tf.reshape(image, [*DIM, CONF["channels"]])
    if mask is not None:
        mask = tf.reshape(mask, [*DIM, 9])
    return image, mask

def CutOut(image, mask=None, DIM=CONF['img_size'], PROBABILITY=0.6, CT=5, SZ=0.1):
    P = tf.cast(random_float() < PROBABILITY, tf.int32)
    if (P == 0) | (CT == 0) | (SZ == 0):
        return image, mask
    for k in range(CT):
        x = tf.cast(tf.random.uniform([], 0, DIM[1]), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, DIM[0]), tf.int32)
        WIDTH = tf.cast(SZ * min(DIM), tf.int32) * P
        ya = tf.math.maximum(0, y - WIDTH // 2)
        yb = tf.math.minimum(DIM[0], y + WIDTH // 2)
        xa = tf.math.maximum(0, x - WIDTH // 2)
        xb = tf.math.minimum(DIM[1], x + WIDTH // 2)
        one = image[ya:yb, 0:xa, :]
        two = tf.zeros([yb - ya, xb - xa, CONF["channels"]], dtype=image.dtype)
        three = image[ya:yb, xb : DIM[1], :]
        middle = tf.concat([one, two, three], axis=1)
        image = tf.concat([image[0:ya, :, :], middle, image[yb : DIM[0], :, :]], axis=0)
        image = tf.reshape(image, [*DIM, CONF["channels"]])
        if mask is not None:
            one = mask[ya:yb, 0:xa, :]
            two = tf.zeros([yb - ya, xb - xa, 9], dtype=mask.dtype)
            three = mask[ya:yb, xb : DIM[1], :]
            middle = tf.concat([one, two, three], axis=1)
            mask = tf.concat([mask[0:ya, :, :], middle, mask[yb : DIM[0], :, :]], axis=0)
            mask = tf.reshape(mask, [*DIM, 9])
    return image, mask
    
def RandomFlip(img, msk=None, hflip_p=0.5, vflip_p=0.5):
    if random_float() < hflip_p:
        img = tf.image.flip_left_right(img)
        if msk is not None:
            msk = tf.image.flip_left_right(msk)
    if random_float() < vflip_p:
        img = tf.image.flip_up_down(img)
        if msk is not None:
            msk = tf.image.flip_up_down(msk)
    return img, msk

def decode_image(data):
    image = tf.io.decode_raw(data, out_type=tf.int16)
    image = tf.reshape(image, (*CONF['img_size'], CONF['channels']))
    image = (image-CONF['a_min']) / (CONF['a_max']-CONF['a_min']) * 255
    return image

def decode_mask(data):    
    mask = tf.io.decode_raw(data, out_type=tf.int8)
    mask = tf.reshape(mask, CONF['img_size'])
    mask = tf.one_hot(mask, depth=9)
    return mask

def read_labeled_tfrecord(example, augment=True):
    LABELED_TFREC_FORMAT = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'mask_raw': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    img = decode_image(example['image_raw'])
    msk = decode_mask(example['mask_raw'])
    if augment:
        img, msk = ShiftScaleRotate(img, msk, DIM=CONF['img_size'], p=0.75)
        img, msk = RandomFlip(img, msk, hflip_p=CONF['hflip'], vflip_p=CONF['vflip'])
        img, msk = CutOut(
            img,
            msk,
            DIM=CONF['img_size'],
            PROBABILITY=CONF['drop_prob'],
            CT=CONF['drop_cnt'],
            SZ=CONF['drop_size'],
        )
    return img, msk

def get_dataset(paths, training=True, augment=True):
    dataset = tf.data.TFRecordDataset(paths, num_parallel_reads=AUTO)
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.cache()
    if training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(1024)
    dataset = dataset.map(
        lambda x: read_labeled_tfrecord(x, augment=augment),
        num_parallel_calls=AUTO
    )
    dataset = dataset.batch(CONF['batch_size'], drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset

cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=CONF['lr'],
    decay_steps=(CONF['epochs']+4) * SET_SIZE['train_set'] // CONF['batch_size'],
    alpha=1e-2
)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=cosine_decay_scheduler)

# Load model
with strategy.scope():
    model = sm.Unet(
        'efficientnetb6',
        classes=9,
        input_shape=(*CONF['img_size'], CONF['channels']),
        encoder_weights='imagenet',
        activation='softmax'
    )
    model.compile(
        optimizer=optimizer,
        loss=sm.losses.DiceLoss(),
        metrics=[sm.metrics.FScore()],
    )
    if CONF['wandb_on']:
        wandb.init(project="CouinaudSegmentation", entity="wandb", id=WANDB_KEY)
        wandb.config.update(CONF)
        callbacks = [
            WandbMetricsLogger(),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_fscore',
                save_best_only=True,
                mode='max',
                save_weights_only=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_fscore',
                factor=0.3,
                patience=5,
                min_lr=1e-6,
                mode='max',
            )
        ]
    else:
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_fscore',
                save_best_only=True,
                mode='max',
                save_weights_only=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_fscore',
                factor=0.3,
                patience=5,
                min_lr=1e-6,
                mode='max',
            )
        ]

# Train model
train_dataset = get_dataset(TRAIN_SET, augment=True)
dev_dataset = get_dataset(DEV_SET, training=False, augment=False)
history = model.fit(
    train_dataset,
    epochs=CONF['epochs'],
    steps_per_epoch=SET_SIZE['train_set'] // CONF['batch_size'],
    validation_data=dev_dataset,
    validation_steps=SET_SIZE['dev_set'] // CONF['batch_size'],
    callbacks=callbacks
)

# Evaluation
test_dataset = get_dataset(TEST_SET, training=False, augment=False)
test_results = model.evaluate(
    test_dataset,
    steps=SET_SIZE['test_set'] // CONF['batch_size'],
)
print("Test FScore:", test_results[1])
tf.keras.models.save_model(model, 'model.h5')
