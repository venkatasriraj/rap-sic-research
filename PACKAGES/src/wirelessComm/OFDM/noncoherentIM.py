"""
Implementation of Noncoherent OFDM-IM
G: Number of clusters 
N: # sub-carriers
L = NG; each block, there are G clusters of N subcarriers
Q: Active subcarriers per cluster
Nb = G*ceil(log2(N combinatorial Q)); number of bits which can be transmitted

"""
import numpy as np
import math
from .ofdm import OFDM

class NoncoherentOFDMIM(OFDM):

    def __init__(self, N, cpLen, G, Q, threshold, A=1):
        super().__init__(N=N*G, cpLen=cpLen)
        self.N = N
        self.cpLen = cpLen
        self.G = G
        self.Q = Q
        self.L = N * G
        self.threshold = threshold
        self.bits_cluster = int(np.floor( math.log2( math.comb(N, Q) ) ))
        self.A = A

    def encodeSubcarriers(self, msg):
        dec = self.bin2dec(msg)
        # print(msg, dec)
        sumSC, k = 0, self.Q
        subcarriers = []
        for i in range(self.N-1, -1, -1):
            temp = math.comb(i, k)
            # print(f"Temp: {temp}, Sum:{sumSC}, dec: {dec}, comb:{temp}")
            if temp+sumSC <= dec:
                subcarriers.append(i)
                sumSC += temp
                k -= 1
        return np.array(subcarriers, dtype=int)[::-1]

    def decodeSubcarriers(self, subcarriers):
        total, k = 0, self.Q
        # print(f"RX: {subcarriers}")
        for sc in subcarriers:
            total += math.comb(sc, k)
            k -= 1
        decodedDec = self.dec2bin(total)
        msgEst = np.concatenate(([0]*(self.bits_cluster-len(decodedDec)), decodedDec), axis=None).astype(int)
        # print(f"Es: {msgEst}")
        if len(msgEst) == self.bits_cluster:
            return msgEst   
        return msgEst[-self.bits_cluster:]

    def transmitModule(self, msg):
        activeSC = np.array([], dtype=int)
        for i in range(self.G):
            clusterSC = self.encodeSubcarriers(msg[i*self.bits_cluster:self.bits_cluster*(i+1)]) + i*self.N
            # print( f"Active SC in cluster{clusterSC-i*self.N}, Msg: {msg[i*self.bits_cluster:self.bits_cluster*(i+1)]}" )
            activeSC = np.concatenate((activeSC, clusterSC), axis=None)
        symbols = np.zeros(self.L, dtype=complex)
        symbols[activeSC] = self.A
        return self.txModule(symbols)

    def receiveModule(self, symbol):
        symbolFreq = self.rxModule(symbol)
        msgEst = np.array([])
        for i in range(self.G):
            clusterList = symbolFreq[i*self.N: (i+1)*self.N]
            activeSC = np.sort( np.argsort(np.abs(clusterList))[-self.Q:] )[::-1]
            # print(f"Len {len(self.decodeSubcarriers(activeSC))}, {activeSC}")
            msgEst = np.concatenate((msgEst, self.decodeSubcarriers(activeSC)), axis=None).astype(int)
        return msgEst

    def receiveModule_1(self, symbol):
        OFDMsymbol = self.rxModule(symbol)
        activeSCEst = np.where(np.abs(OFDMsymbol) >= self.threshold)[0]
        msgEst = np.array([])
        for i in range(self.G):
            # print(i, i*self.N, (i+1)*self.N)
            cluterSCEst = np.asarray(list( filter(lambda x: i*self.N <= x < (i+1)*self.N, list(activeSCEst)) )) - i * self.N
            # print(cluterSCEst)
            msgEst = np.concatenate((msgEst, self.decodeSubcarriers(cluterSCEst)), axis=None).astype(int)
        # print(msgEst)
        return msgEst