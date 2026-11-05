import numpy as np

class HelperFunctions:

    @staticmethod
    def bin2dec(binData):
        if len(binData) == 0:
            return 0
        power2 = 2**np.arange(len(binData))[::-1]
        return int(np.sum(binData * power2))

    @staticmethod
    def dec2bin(decData):
        binData = [decData % 2]
        decData //= 2
        while decData>0:
            binData += [decData%2]
            decData //= 2
        return np.asarray(binData)[::-1]

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