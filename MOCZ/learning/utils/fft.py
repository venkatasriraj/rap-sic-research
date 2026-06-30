import numpy as np

def fftCon(y, K, Q, R):
    N_r = len(y)
    N_fft = Q * K

    scaling_vec = R ** np.arange(N_r)
    y_ctr = np.conjugate(y[::-1])

    y_scaled = y * scaling_vec
    y_ctr_scaled = y_ctr * scaling_vec

    y_pad = np.pad(y_scaled, (0, N_fft - N_r), mode='constant')
    y_ctr_pad = np.pad(y_ctr_scaled, (0, N_fft - N_r), mode='constant')

    Y_eval = np.abs( np.fft.ifft(y_pad) )
    Y_ctr_eval = np.abs( np.fft.ifft(y_ctr_pad) )

    return Y_eval, Y_ctr_eval