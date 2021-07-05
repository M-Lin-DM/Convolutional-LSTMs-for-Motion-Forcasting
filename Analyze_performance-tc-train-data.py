import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import plotly.graph_objects as go
import plotly.express as px

model = keras.models.load_model(r"C:\Users\MrLin\Documents\Experiments\ConvLSTM_forcasting\checkpoint_tc-augm-lastepoch-retarget-noclassweights")

Y_label = np.load(r'D:\Datasets\Fish\processed\targets\train\Y_label.npy')
print("shape: {x.shape}, dtype: {x.dtype}".format(x=Y_label))

# Build test dataset generator
# define parameters for this dataset
fps = 30  # frames per second. the input tensor will cover 4*n/fps seconds
T = len(Y_label)  # number of images in the dataset
batch_size = 4
n = 30  # the length of the time axis of the model's input tensor. 4n video frames are divided up to construct n tensor slices
h = 10  # look-ahead. we average the consecutive frame difference over the h+1 next frames, including the current
tc_start_frame = 4 * n - 1  # first frame inded in the valid range
tc_end_frame = T - 2 - h  # last frame index in the valid range, assuming images start at t=0 and go to t=T-1

target_size = (120, 160)
clip_tensor_shape = (n,) + tuple(target_size) + (4,)
steps_per_epoch_tuning = 100 #1250
training_dir = "D:/Datasets/Fish/processed/train/New folder/"


def tc_inputs_generator(t_start, t_end, image_folder):
    t = t_start
    while t <= t_end:
        clip_tensor = []
        for j in range(t - 4 * n + 1, t + 1, 1):
            pil_img = load_img(image_folder.decode('UTF-8') + f'{j:05}' + '.png', color_mode='grayscale',
                               target_size=target_size)
            clip_tensor.append(
                img_to_array(pil_img, data_format='channels_last', dtype='uint8').astype(np.float32) / 255)
        clip_tensor = np.array(clip_tensor)  # concat all 4n clip frames along time axis
        clip_tensor = np.transpose(clip_tensor,
                                   axes=(3, 1, 2, 0))  # permute dims so tensor.shape = (1, height, width, 4n)
        clip_tensor = np.array_split(clip_tensor, n,
                                     axis=3)  # returns a list of n tensors, each with shape=(1, height, width, 4)
        clip_tensor = np.concatenate(clip_tensor,
                                     axis=0)  # finally concat along time axis, producing shape (n, height, width, 4)
        yield clip_tensor
        t += 1


# formula to get cardinality of the batched dataset if .cardinality wont work
print("number of samples in dataset: ", (tc_end_frame - tc_start_frame + 1) / 1)

ds_train = tf.data.Dataset.from_generator(
    tc_inputs_generator,
    args=[tc_start_frame, tc_end_frame, training_dir],
    output_types=(tf.float32),
    output_shapes=clip_tensor_shape)

print(ds_train.element_spec)

ds_train = ds_train.batch(batch_size, drop_remainder=False)
ds_train = ds_train.take(steps_per_epoch_tuning)

# PREDICT USING MODEL
Y_pred = model.predict(ds_train)
np.save(r'D:\Datasets\Fish\processed\targets\train\Y_pred', Y_pred)

Y_tilde = np.load(r'D:\Datasets\Fish\processed\targets\train\Y_tilde.npy')
Y_pred = np.load(r'D:\Datasets\Fish\processed\targets\train\Y_pred.npy')


def pad_y_pred(Y, T, start, end):
    out = np.zeros(T)
    out[start:end] = Y
    return out

# crop data to valid range
def crop_to_valid_range(Y, n, h):
    T = len(Y)
    start_frame = 4 * n - 1  # first frame inded in the valid range
    end_frame = T - 2 - h
    return Y[start_frame:end_frame + 1]


print("number of samples in dataset: ", (tc_end_frame - tc_start_frame + 1) / 1)
print(f"crop_to_valid_range(Y_label, n, h).shape: {crop_to_valid_range(Y_label, n, h).shape}, y_pred.shape: {Y_pred.shape}")

# NOTE: y_pred may not have the same length as crop_to_valid_range(Y_label, n, h). The reason is drop_remainder was set to True during batching
# Therefore we need to take elements [:len(y_pred)] in the lines below. if predicting over a truncated training set, we must use [:len(Y_pred)] because y_pred is much shorter than y_label
fpr, tpr, thresholds = roc_curve(crop_to_valid_range(Y_label, n, h)[0:len(Y_pred)+0], Y_pred[:, 0], drop_intermediate=True) # to shift Y_label backward by 70 frames, [70:len(Y_pred)+70]
auc = roc_auc_score(crop_to_valid_range(Y_label, n, h)[:len(Y_pred)], Y_pred[:, 0])
print(f"auc: {auc}")


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc, linestyle=':')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# Plot timeseries-------------------
y_pred_padded = pad_y_pred(Y_pred[:,0], T, tc_start_frame, tc_start_frame+len(Y_pred)) # we dont use end_frame because Y_pred is going to be short. That is the whole point of this file: to avoid predicting over the entire dataset
print(f"y_pred_padded.shape: {y_pred_padded.shape}")

x = np.arange(0, len(Y_label), 1)
start_frame = tc_start_frame
end_frame = tc_start_frame + len(Y_pred)

# Create traces
fig = go.Figure()

fig.add_trace(go.Scatter(name='ytilde',
                         x=x[start_frame:end_frame], y=Y_tilde[start_frame:end_frame],
                         mode='lines',

                         line=dict(
                             width=1,
                             color='gray'
                         ),
                         showlegend=False

                         ))

fig.add_trace(go.Scatter(name='ConvLSTM prediction',
                         x=x[start_frame:end_frame], y=Y_tilde[start_frame:end_frame],
                         mode='markers',

                         marker=dict(
                             color=y_pred_padded[start_frame:end_frame],
                             size=10,
                             opacity=0.7,
                             colorscale="Jet",
                             line=dict(
                                 color='black',
                                 width=1
                             )
                         ),

                         ))

fig.add_trace(go.Scatter(name='positive (ground truth)',
                         x=x[Y_label == 1], y=Y_tilde[Y_label == 1],
                         mode='markers',

                         marker=dict(
                             size=5,
                             color='hotpink'
                         ),

                         ))

fig.update_layout(
    width=900,
    height=700,

    legend=dict(
        yanchor="top",
        y=0.98,
        xanchor="left",
        x=0.01,
        font_size=14
    ),
    xaxis=dict(title_text='frames', title_font_size=18),
    yaxis=dict(title_text='\u1EF9', title_font_size=18)

)

fig.show()

# histogram of Y_pred
print(np.mean(y_pred_padded[start_frame:end_frame]))
# fig = go.Figure(data=[go.Histogram(x=y_pred, cumulative_enabled=False, histnorm='probability')])
fig2 = px.histogram(Y_pred, nbins=50)
fig2.show()

print(y_pred_padded[start_frame:end_frame])