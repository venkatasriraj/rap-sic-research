import numpy as np

class LoRa:

    def __init__(self, SF, BW, sampRate, initialFreq):
        self.SF = SF
        self.BW = BW
        self.sampRate = sampRate
        self.symDuration = 2**SF / BW
        self.chirpRate = BW / self.symDuration
        self.initialFreq = initialFreq

    @staticmethod
    def bin2dec(binArr):
        pow2 = 2**np.arange(len(binArr))
        return np.sum( np.flip(binArr) * pow2 )

    def dec2bin(self, dec):
        bin = np.array([dec%2])
        while dec//2 !=0:
            dec //= 2
            bin = np.append(bin, dec%2)   
        bin = np.append(bin, [0]*(self.SF - len(bin)))
        return np.flip(bin)
    
    def genRefChirp(self):
        time = np.arange(0, self.symDuration, 1/self.sampRate)
        phase_t = 2*np.pi* ( self.initialFreq*time + 0.5*self.chirpRate * time**2 )
        return np.exp( 1j * phase_t )

    def genSymChirp(self, refChirp, symbol):
        sampPerChirp = len(refChirp) / 2**self.SF
        return np.roll(refChirp, int(-symbol * sampPerChirp))

    def modulation(self, msg):
        rem = len(msg) % self.SF
        if rem != 0:
            msg = np.append(msg, [0]*(self.SF - rem))
        chirp = np.array([])
        refChirp = self.genRefChirp()
        for i in range(len(msg)//self.SF):
            symbol = self.bin2dec(msg[i*self.SF : (i+1) * self.SF])
            chirp = np.append(chirp, self.genSymChirp(refChirp, symbol))
        return chirp

    def demodulation(self, rxChirp):
        refChirp = self.genRefChirp()
        numSamples = len(refChirp)
        msg_decoded = []
        reminder = len(rxChirp) % numSamples
        addZeros = 0 if reminder == 0 else numSamples-reminder
        rxChirp = np.append(rxChirp, [0]*addZeros)
        numSym = len(rxChirp) // numSamples
        for i in range(numSym):
            symChirp = rxChirp[i*numSamples: (i+1)*numSamples]
            # if len(symChirp) != numSamples:
            #     symChirp = np.append(symChirp, [0]*(numSamples - len(symChirp)))
            dechirp = symChirp * np.conj(refChirp)
            fftTone = np.fft.fft(dechirp)
            symbol_hat = np.argmax( np.abs(fftTone) )
            msg_decoded = np.append(msg_decoded, self.dec2bin(symbol_hat))
        return msg_decoded.astype(np.uint16)