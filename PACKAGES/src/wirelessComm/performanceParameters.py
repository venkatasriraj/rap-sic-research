import numpy as np

class PerformanceParameters:

    @staticmethod
    def pcr(msgRx, msgTx):
        # -- packet correct rate
        if np.all( msgRx == msgTx ):
            return 1
        else:
            return 0

    @staticmethod
    def mae(sig_rx, sig_tx):
        return np.mean( np.abs(sig_rx - sig_tx) )

    @staticmethod
    def bitError(msgRx, msgTx):
        return np.sum(msgRx != msgTx)

    @staticmethod
    def ber(msgRx, msgTx):
        return np.mean( msgRx != msgTx )

    @staticmethod
    def errorPos(msgRx, msgTx):
        return np.where(np.asarray(msgRx) != np.asarray(msgTx))

    @staticmethod
    def PAPR(signal):
        # signal_max = np.abs( np.sum(signal) )
        signal_max = np.max( np.abs(signal) )**2
        signal_power = np.mean( np.abs(signal)**2 )
        papr = signal_max / signal_power
        papr_dB = 10*np.log10(papr)
        # print(f"Signal Power: {signal_power}, PAPR: {papr}, Signal Energy: "
        #         f"{signal_power * len(signal)}, Max Abs Coeff: {np.max(np.abs(signal))}")
        return papr_dB