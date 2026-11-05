"""
DE: differential encoding based modultation and demodulation
(in this file it is known as bitCoder)

DPSK - differential modulation: data is encoded in the phase difference between consecutive symbols
"""

import numpy as np

class DPSK:

    def __init__(self):
        self.refSym = 1

    def bitCoderModulation(self, bits):
        encodedBits = self.differentialEncoding(bits)
        return np.array(2*encodedBits-1).astype(np.complex64)

    def bitCoderDemodulate(self, samples):
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples)
        bits = (samples.real > 0).astype(int)
        decodedBits = self.differentialDecoding(bits)
        return decodedBits

    @staticmethod
    def differentialEncoding(bits):
        encodedBits = []
        prevBit = 0
        encodedBits.append(prevBit)
        for i in bits:
            prevBit = i ^ prevBit
            encodedBits.append(prevBit)
        return np.array(encodedBits)

    @staticmethod
    def differentialDecoding(bits):
        decodedBits = []
        previousBit = bits[0]
        for i in range(1, len(bits)):
            decodedBits.append(bits[i] ^ previousBit)
            previousBit = bits[i]
        return np.array(decodedBits)

    def differentialDemodulation_1(self, symbols):
        decoded = []
        for i in range(len(symbols)-1):
            bit = 1 if np.real( symbols[i+1] * np.conj(symbols[i]) ) < 0 else 0
            decoded += [bit]
        return np.array(decoded)

    def differentialModulation(self, bits):
        symbols = [self.refSym]
        prevSym = self.refSym
        for bit in bits:
            if bit == 0:
                SYMBOL = prevSym
            else:
                SYMBOL = prevSym * np.exp(1j*np.pi)
            symbols.append(SYMBOL)
            prevSym = SYMBOL
        return np.asarray(symbols, dtype=complex)

    def differentialDemodulation(self, symbols):
        msgRx = []
        for i in range(len(symbols)-1):
            phase = np.angle(symbols[i+1] * np.conjugate(symbols[i]))
            if -np.pi/2 < phase < np.pi/2:
                msgRx.append(0)
            else:
                msgRx.append(1)
        return msgRx

    def CFOestimation(self, symbols):
        phi_hat = 0
        for i in range(len(symbols)-1):
            z = symbols[i+1] * np.conjugate(symbols[i])
            phi_hat += (z**2)
        return np.angle(phi_hat) / ( 2 * (len(symbols)-1) )

    def transmitterRAP(self, msgTx, ENERGY):
        sigTx = self.differentialModulation(msgTx)         
        l2_norm = np.linalg.norm(sigTx, ord=2) 
        sigTx = (sigTx / l2_norm) * np.sqrt(ENERGY)
        return sigTx

    def receiverRAP(self, sigRx, CFO='False'):
        rotation_hat = 0
        if CFO:
            rotation_hat = self.CFOestimation(sigRx)
        msgRx = self.differentialDemodulation(sigRx)
        return msgRx, rotation_hat