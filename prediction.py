from dataset import covid_dataset, extract_info
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

print(tf.__version__)
model = tf.keras.models.load_model('test_model.h5')
datas = covid_dataset.confirmed_case_per_day
for i in range(10):
    datas.append(0)
print(datas)
series = np.array(datas)
current_time = 333
time_valid = range(current_time, len(datas))
x_valid = series[current_time:]


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

window_size = 5
rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[current_time - window_size:-1, -1, 0]
print(rnn_forecast)
# print(tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy())

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)
plt.show()