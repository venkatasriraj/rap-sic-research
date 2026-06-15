from utils import fftCon
import numpy as np

def fftDizet(y, K, Q, R):
    Y_eval, Y_ctr_eval = fftCon(y, K, Q, R)

    message_received = ( 1 - np.sign( Y_eval[::Q] - Y_ctr_eval[::Q] ) ) / 2 

    return message_received