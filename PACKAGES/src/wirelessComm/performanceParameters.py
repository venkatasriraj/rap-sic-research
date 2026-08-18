import numpy as np

class PerformanceParameters:

    @staticmethod
    def pcr(msg_rx, msg_tx):
        # -- Packet correct rate
        if np.all( msg_rx == msg_tx ):
            return 1
        else:
            return 0

    @staticmethod
    def mae(sig_rx, sig_tx):
        return np.mean( np.abs(sig_rx - sig_tx) )

    @staticmethod
    def ber(msg_rx, msg_tx):
        return np.mean( msg_rx != msg_tx )