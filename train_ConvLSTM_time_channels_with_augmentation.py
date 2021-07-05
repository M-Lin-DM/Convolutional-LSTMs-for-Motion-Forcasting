import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.layers import Input, ConvLSTM2D, Dense, TimeDistributed, Flatten
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.models import Model

import wandb
from wandb.keras import WandbCallback

wandb.init(entity='your_wanby_ID', project='ConvLSTM_fish', group='tc-exp-1',
           config={"n": 30,
                   "h": 45,
                   "batch_size": 4,
                   "target_size": (120, 160),  # (img_height, img_width)
                   "epochs": 5,
                   "dropout": 0.0,
                   "L1": 0.000,
                   "L2": 0.000})
config = wandb.config

n = config.n  # the length of the time axis of the model's input tensor. 4n video frames are divided up to construct n tensor slices
h = config.h  # look-ahead. we average the consecutive frame difference over the h+1 next frames, including the current

# tc_clip_length = len([j for j in range(1000 - 4 * n + 1, 1000 + 1, 1)])  # the number of frames used to produce the video segment fed to the ConvLSTM.
# assert tc_clip_length == 4 * n
batch_size = config.batch_size
target_size = config.target_size
clip_tensor_shape = (n,) + tuple(target_size) + (4,)  # notice the input tensor has length n, not tc_clip_length
steps_per_epoch_tuning = 100 #11230  # fraction the dataset's total batches (11231.5 batches, batchsize=4) or (5615.75 batchsize=8)
print(f"clip_tensor_shape: {clip_tensor_shape} | steps_per_epoch_tuning: {steps_per_epoch_tuning}")

training_dir = 'D:/Datasets/Fish/processed/train/New folder/'
validation_dir = 'D:/Datasets/Fish/processed/validation/New folder/'

# Create tf.Data.datasets generators----------------
def tc_inputs_generator(t_start, t_end, image_folder, targets):
    t = t_start
    while t <= t_end:
        clip_tensor = []
        for j in range(t - 4 * n + 1, t + 1, 1):
            pil_img = load_img(image_folder.decode('UTF-8') + f'{j:05}' + '.png', color_mode='grayscale',
                               target_size=target_size)
            clip_tensor.append(
                img_to_array(pil_img, data_format='channels_last', dtype='uint8').astype(np.float32) / 255)
        clip_tensor = np.array(
            clip_tensor)  # concat all 4n clip frames along time axis. output shape = (4n, height, width, 1)
        clip_tensor = np.transpose(clip_tensor,
                                   axes=(3, 1, 2, 0))  # permute dims so tensor.shape = (1, height, width, 4n)
        clip_tensor = np.array_split(clip_tensor, n,
                                     axis=3)  # returns a list of n tensors, each with shape=(1, height, width, 4)
        clip_tensor = np.concatenate(clip_tensor,
                                     axis=0)  # finally concat along time axis, producing shape (n, height, width, 4)
        yield clip_tensor, targets[t]
        t += 1


# Load ground truth labels and define parameters for train dataset
Y_label_train = np.load(r'D:\Datasets\Fish\processed\targets\train\Y_label.npy')
Y_label_train = np.array([int(Y_label_train[j]) for j in range(len(Y_label_train))])
T_train = len(Y_label_train)  # number of images in the dataset
start_frame_tr = 4 * n - 1  # first frame inded in the valid range
end_frame_tr = T_train - 2 - h  # last frame index in the valid range, assuming images start at t=0 and go to t=T-1
samples_tr = end_frame_tr - start_frame_tr + 1
steps_per_epoch_tr = int(np.floor((end_frame_tr - start_frame_tr + 1) / config.batch_size))  # number of batches

Y_label_validation = np.load(r'D:\Datasets\Fish\processed\targets\validation\Y_label.npy')
Y_label_validation = np.array([int(Y_label_validation[j]) for j in range(len(Y_label_validation))])
T_validation = len(Y_label_validation)  # number of images in the dataset
start_frame_val = 4 * n - 1  # first frame inded in the valid range
end_frame_val = T_validation - 2 - h  # last frame index in the valid range, assuming images start at t=0 and go to t=T-1
steps_per_epoch_val = int(np.floor((end_frame_val - start_frame_val + 1) / config.batch_size))

ds_train = tf.data.Dataset.from_generator(
    tc_inputs_generator,
    args=[start_frame_tr, end_frame_tr, training_dir, Y_label_train],
    output_types=(tf.float32, tf.int32),
    output_shapes=(clip_tensor_shape, ()))

ds_validation = tf.data.Dataset.from_generator(
    tc_inputs_generator,
    args=[start_frame_val, end_frame_val, validation_dir, Y_label_validation],
    output_types=(tf.float32, tf.int32),
    output_shapes=(clip_tensor_shape, ()))

ds_validation = ds_validation.batch(batch_size, drop_remainder=True).repeat(config.epochs)

# Data Augmentation: we flip and rotate each element of the dataset using dataset.map(rotation function)-----------------
# ds_train_identity = ds_train # i dont think this is usable because if ds_train is modified, ds_train_identity may be modified. NO the reason was because I mistakenly multiplied the steps_per_epoch
ds_train_flip_HV = ds_train.map(lambda x, targ: (tf.image.flip_up_down(tf.image.flip_left_right(x)), targ))
ds_train_flip_H = ds_train.map(lambda x, targ: (tf.image.flip_left_right(x), targ))
ds_train_flip_V = ds_train.map(lambda x, targ: (tf.image.flip_up_down(x), targ))

# concat datasets:
ds_train = ds_train.concatenate(ds_train_flip_HV).concatenate(ds_train_flip_H).concatenate(ds_train_flip_V)
# ---------NOTE: COMMENT OUT LINE BELOW IF NOT USING DATA AUGMENTATION---------
steps_per_epoch_tr *= 4  # multiply the number of batches by 4 since we did a 4x augmentation

# inspect
print(ds_train.element_spec)
for x, y in ds_train.take(3):
    print('shapes: {x.shape},{y.shape}'.format(x=x, y=y))

# print('shuffling dataset')
ds_train = ds_train.shuffle(int(samples_tr * 0.04),
                            # turn off shuffle if you are training on a subset of the data for hyperparameter tuning and plan to visualize performance on the first steps_per_epoch_tuning batches of the TRAINING set
                            reshuffle_each_iteration=False)  # the argument into shuffle is the buffer size. this can be smaller than the number of samples, especially when using larger datasets
ds_train = ds_train.batch(batch_size, drop_remainder=True)  # insufficient data error was thrown without adding the .repeat(). I added the +1 dataset at the end for good measure
ds_train = ds_train.repeat(config.epochs + 1)

# Below is for training on a subset of the data during hyperparameter tuning/prototyping------------------
# ds_train = ds_train.take(steps_per_epoch_tuning).repeat(
#     config.epochs+1)  # 600 batches of batch_size =8 is about 10% of a dataset with 44000 samples ie (~5488 batches)

print("shapes after batching", ds_train.element_spec)
# # Build model--------------------------
pooling_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid', data_format='channels_last')
batch_norm_1_layer = tf.keras.layers.BatchNormalization(axis=-1, center=False, scale=False, name='batchnorm_1')
batch_norm_2_layer = tf.keras.layers.BatchNormalization(axis=-1, center=False, scale=False, name='batchnorm_2')
batch_norm_3_layer = tf.keras.layers.BatchNormalization(axis=-1, center=False, scale=False, name='batchnorm_3')
L1L2 = tf.keras.regularizers.L1L2(l1=config.L1, l2=config.L2)

inputs = Input(shape=clip_tensor_shape)
convLSTM_1 = ConvLSTM2D(16, 3, strides=(1, 1), padding='valid', data_format='channels_last', activation='tanh',
                        recurrent_activation='hard_sigmoid', use_bias=False, return_sequences=False, return_state=False,
                        dropout=config.dropout, kernel_regularizer=L1L2,
                        name='convLSTM_1')(inputs)
batch_norm_0 = tf.keras.layers.BatchNormalization(axis=-1, center=False, scale=False, name='batchnorm_1')(convLSTM_1) # only used if return_sequences=False in first convLSTM layer. ie only one convlstm layer
# max_pool_1 = TimeDistributed(pooling_layer)(convLSTM_1)
# batch_norm_1 = TimeDistributed(batch_norm_1_layer)(max_pool_1)

# convLSTM_2 = ConvLSTM2D(64, 3, strides=(1, 1), padding='valid', data_format='channels_last', activation='tanh',
#                         recurrent_activation='hard_sigmoid', use_bias=False, return_sequences=False, return_state=False,
#                         dropout=config.dropout, kernel_regularizer=L1L2,
#                         name='convLSTM_2')(batch_norm_1)
# # max_pool_2 = TimeDistributed(pooling_layer)(convLSTM_2)
# # batch_norm_2 = TimeDistributed(batch_norm_2_layer)(max_pool_2)
# #
# # convLSTM_3 = ConvLSTM2D(64, 3, strides=(1, 1), padding='valid', data_format='channels_last', activation='tanh',
# #                         recurrent_activation='hard_sigmoid', use_bias=False, return_sequences=False, return_state=False,
# #                         dropout=config.dropout, kernel_regularizer=L1L2,
# #                         name='convLSTM_3')(batch_norm_2)
# # max_pool_3 = TimeDistributed(pooling_layer)(convLSTM_3)
# # batch_norm_3 = TimeDistributed(batch_norm_3_layer)(max_pool_3)
# #
# # convLSTM_4 = ConvLSTM2D(128, 1, strides=(1, 1), padding='valid', data_format='channels_last', activation='tanh',
# #                         recurrent_activation='hard_sigmoid', use_bias=True, return_sequences=False, return_state=False,
# #                         name='convLSTM_4')(batch_norm_3)
# # opt 1: flattening preserves information about the spatial locations of detected features. This may turn out to be excessive information for the classifier..
# # max_pool_4 = pooling_layer(convLSTM_4) # you could also try global max pooling instead of pooling+flatten
# # z = Flatten(data_format='channels_last', name='flatten')(max_pool_4)
#
# # opt 2: the rationale is that we only want to detect the presence* (not location) of features using the last convlstm's filters. We may not care about the spatial locations of these features in image
z = tf.keras.layers.GlobalMaxPool2D(data_format='channels_last')(batch_norm_0)
#tested modification: adding more dense layers
# z2 = Dense(16, activation='sigmoid')(z)
# z3 = Dense(8, activation='sigmoid')(z2)
y = Dense(1, activation='sigmoid')(z)
#
model = Model(inputs, y)
#
L = tf.keras.losses.BinaryCrossentropy(name='binary_crossentropy_in_loss') # NOTE: when using class weights each sample-wise loss will be weighted by the sample's class weight. Thus the reported 'loss' will be lower than the 'metric' using the same function.
model.compile(loss=L,
              optimizer=keras.optimizers.Adam(learning_rate=0.00008, beta_1=0.9, beta_2=0.999, amsgrad=False),
              metrics=[tf.keras.metrics.BinaryCrossentropy(name='binary_crossentropy_in_metrics'), tf.keras.metrics.AUC()])

# END build model----------------

# Load model if you are resuming training. (It takes a long time..) In that case, Do NOT recompile model or you will lose the optimizer state.------------
# model = keras.models.load_model(r"C:\Users\MrLin\Documents\Experiments\ConvLSTM_forcasting\checkpoint_tc-augm-lastepoch-retarget-no-regr")
# model.summary()
class_weight = {0: 0.133, 1: 0.866}  # the training data has 86.6% negative samples

# # Train model-------------------------------
checkpoint_path = r'C:\Users\MrLin\Documents\Experiments\ConvLSTM_forcasting\checkpoint_tc-augm-lastepoch-retarget-no-regr-single-lstm'
checkpoint_path_best_validation = r'C:\Users\MrLin\Documents\Experiments\ConvLSTM_forcasting\checkpoint_tc-augm-best-validation-retarget-single-lstm'
# Make checkpoint callback
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    monitor='loss', #the loss, weighted by class weights
    # change to 'loss' instead of 'val_loss' if youre tuning hyperparams and training on a small subset of the data
    mode='min',
    save_best_only=False,
    save_freq='epoch')  # change to false if youre tuning hyperparams and training on a small subset of the data

model_checkpoint_callback_best_validation = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path_best_validation,
    save_weights_only=False,
    monitor='val_loss',
    # change to 'loss' instead of 'val_loss' if youre tuning hyperparams and training on a small subset of the data
    mode='min',
    save_best_only=True)  # change to false if youre tuning hyperparams and training on a small subset of the data

model.fit(ds_train,
          # shuffle=True, NOTE: "This [shuffle] argument is ignored when x is a generator or an object of tf.data.Dataset"
          steps_per_epoch=steps_per_epoch_tr,
          # 5607,  # steps_per_epoch needs to equal exactly the number of batches in the Dataset generator
          epochs=config.epochs,
          verbose=2,
          validation_data=ds_validation,
          validation_steps=steps_per_epoch_val,  # 436,  # =#batches in validation dataset
          class_weight=class_weight,
          callbacks=[WandbCallback(), model_checkpoint_callback, model_checkpoint_callback_best_validation])  #
