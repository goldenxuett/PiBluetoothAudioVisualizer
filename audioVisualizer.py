import pyaudio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy import signal
import math

def split_by_index(data, indices):
    binned_data = []
    previous = 0
    for i in range(len(indices)):
        binned_data.append(np.mean(data[previous:indices[i]]))
        previous = indices[i]
    binned_data.append(np.mean(data[previous:len(data)]))
    return binned_data

def filter_low_and_high(s, order=2, frequency = 1000):
    b,a = signal.butter(order, 20.0/(frequency/2.0), btype = 'highpass', analog = False)
    s = signal.filtfilt(b, a, s)
    b,a = signal.butter(order, 70.0/(frequency/2.0), btype = 'lowpass', analog = False)
    s = signal.filtfilt(b, a, s)
    return s

fig = plt.figure(figsize=(5,5))
ax = plt.axes(xlim=(0, 6), ylim=(0, 6))
ax.set_facecolor([0,0,0])
base1 = plt.Circle((1, 1), 0.2, fc= [148.0/255, 0, 211.0/255])
base2 = plt.Circle((1, 3), 0.2, fc= [75.0/255, 0, 130.0/255])
base3 = plt.Circle((1, 5), 0.2, fc= [0,0,1])
base4 = plt.Circle((3, 1), 0.2, fc= [0,1,0])
base5 = plt.Circle((3, 3), 0.2, fc= [1,1,0])
base6 = plt.Circle((3, 5), 0.2, fc= [1, 127.0/255, 0])
base7 = plt.Circle((5, 3), 0.2, fc= [1,0,0])


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 4096

log_scale = []
max_bin = np.log10(CHUNK)
indices = []
num_bins = 7
for i in range(num_bins):
    log_scale.append(max_bin / num_bins * (i+1))

for i in range(len(log_scale) - 1):
    indices.append(math.floor(10**((log_scale[i]*log_scale[i+1])**(0.5))))

ham = np.hamming(CHUNK)
controller = pyaudio.PyAudio()
stream = controller.open(format = FORMAT, channels = CHANNELS, rate = RATE, frames_per_buffer = CHUNK, input = True)

def init():
    ax.add_patch(base1)
    ax.add_patch(base2)
    ax.add_patch(base3)
    ax.add_patch(base4)
    ax.add_patch(base5)
    ax.add_patch(base6)
    ax.add_patch(base7)
    return base1, base2, base3, base4, base5, base6, base7,

def loop(i):
    stream_data = stream.read(CHUNK,exception_on_overflow = False)
    sample = np.fromstring(stream_data, dtype=np.int16).astype(np.float32) # <- @TODO replace this line with code to read a sample from the microphone.
    sample = filter_low_and_high(sample, frequency = CHUNK)
    hammed_data = np.multiply(sample, ham)
    transform = np.fft.fft(hammed_data)
    pst = 1.0 / (CHUNK) * (abs(transform))**2
    binned_pst = split_by_index(pst, indices)
    if sum(binned_pst) > 0: 
        calc_base1 =  0.000005 * binned_pst[0] 
        base1.set_radius(calc_base1)
        calc_base2 =  0.000000005 * binned_pst[1] 
        base2.set_radius(calc_base2)
        calc_base3 =  0.000000005 * binned_pst[2] 
        base3.set_radius(calc_base3)

        calc_base4 =  0.00000005 * binned_pst[3] 
        base4.set_radius(calc_base4)
        calc_base5 =  0.0005 * binned_pst[4] 
        base5.set_radius(calc_base5)
        calc_base6 =  0.0005 * binned_pst[5] 
        base6.set_radius(calc_base6)
        calc_base7 =  0.00000005 * binned_pst[6] 
        base7.set_radius(calc_base7)

        return base1, base2, base3, base4, base5, base6, base7, 


anim = animation.FuncAnimation(fig, loop, 
                               init_func=init, 
                               frames=10, 
                               interval=10,
                               blit=True)

plt.show()
