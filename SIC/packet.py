"""
Designing a packet structure which will be used for RAP.

userId - 1B
frameNo - 1B
degree - 1B
thisSlot - 1B
slotList - 2B
accessCode - 4B
"""

import numpy as np
import binascii
import random

class PacketStructure:

    def __init__(self, maxDegree, AccessCode, pktSize):
        self.accessCode = AccessCode
        self.maxDegree = maxDegree
        self.pktSize = pktSize

    def buildPacket(self, userId, frameNo, degree, thisSlot, slotList):

        userId_b = [ userId & 0xFF ]
        frameNo_b = [frameNo & 0xFF]
        degree_b = [degree & 0xFF]
        thisSlot_b = [thisSlot & 0xFF]
        slotList_b = [s & 0xFF for s in slotList]
        slotList_b += [0x00] * (self.maxDegree - len(slotList))

        protectedData = userId_b + frameNo_b + degree_b + thisSlot_b + slotList_b

        random.seed(userId)
        dataLen = self.pktSize - len(protectedData) - len(self.accessCode)
        data_b = [random.randint(0,255) & 0xFFFF for _ in range(dataLen)]
        # print(f"Length of payload {len(data_b)}")
        protectedData += data_b

        protectedCRC = binascii.crc32( bytes(protectedData) ) & 0xFFFFFFFF
        protectedCRC_b = list( protectedCRC.to_bytes(4, byteorder='big') )
            # ---- input to CRC is bytes and we to get unsigned CRC we will mask with 0xFFFFFFFF
        pkt = self.accessCode + protectedData + protectedCRC_b
        return pkt

    def parsePacket(self, pkt):

        

        return None