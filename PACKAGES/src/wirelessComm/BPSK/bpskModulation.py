import numpy as np

class BPSKBase:

    def modulate(self, bits):
        encodedBits = self.differentialEncoding(bits)
        return np.array(2*encodedBits-1).astype(np.complex64)

    def demodulate(self, samples):
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples)
        bits = (samples.real > 0).astype(int)
        decodedBits = self.differentialDecoding(bits)
        return decodedBits

    @staticmethod
    def bytes_to_bits(pkt):
        bits = []
        for data in pkt:
            for i in range(7,-1,-1):
                bits.append( (data >> i) & 1 )
        return np.array(bits)

    @staticmethod
    def bits_to_bytes(bits):
        bits = np.asarray(bits, dtype=np.uint8)
        n = (len(bits) // 8) * 8
        return [int(''.join( map(str, bits[i:i+8]) ), 2) for i in range(0, n, 8)]

    @staticmethod
    def differentialEncoding(bits):
        encodedBits = []
        prevBit = bits[0]
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