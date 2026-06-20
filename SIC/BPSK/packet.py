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
        self.payloadLen = 0

    def buildPacket(self, userId, frameNo, degree, thisSlot, slotList):

        userId_b = [ userId & 0xFF ]
        frameNo_b = [frameNo & 0xFF]
        degree_b = [degree & 0xFF]
        thisSlot_b = [thisSlot & 0xFF]
        slotList_b = [s & 0xFF for s in slotList]
        slotList_b += [0x00] * (self.maxDegree - len(slotList))

        protectedData = userId_b + frameNo_b + degree_b + thisSlot_b + slotList_b

        random.seed(userId)
        self.payloadLen = self.pktSize - len(protectedData) - len(self.accessCode) - 4
        # 4 Bytes for CRC
        data_b = [random.randint(0,255) & 0xFFFF for _ in range(self.payloadLen)]
        # print(f"Length of payload {len(data_b)}")
        protectedData += data_b

        protectedCRC = binascii.crc32( bytes(protectedData) ) & 0xFFFFFFFF
        protectedCRC_b = list( protectedCRC.to_bytes(4, byteorder='little') )
            # ---- input to CRC is bytes and we to get unsigned CRC we will mask with 0xFFFFFFFF
        pkt = self.accessCode + protectedData + protectedCRC_b
        return pkt

    def parsePacket(self, pkt):
        AC_Len = len(self.accessCode)
        idx = AC_Len
        userId = pkt[idx]; idx += 1
        frameNo = pkt[idx]; idx += 1
        degree = pkt[idx]; idx += 1
        thisSlot = pkt[idx]; idx += 1
        slotList = pkt[idx: idx+degree]; idx += self.maxDegree
        payload = pkt[idx: idx + self.payloadLen]; idx += self.payloadLen
        crc_rx = pkt[idx:idx+4]

        protected = pkt[AC_Len: AC_Len+1+1+1+1+self.maxDegree+self.payloadLen]
        crc_act = binascii.crc32( bytes(protected) ) & 0xFFFFFFFF
        crc_act_b = list( crc_act.to_bytes(4, byteorder='little') )
        return {
            "userId": userId,
            "frameNo": frameNo,
            "degree": degree,
            "thisSlot": thisSlot,
            "slotList": slotList,
            "payload": payload,
            "crc_ok": (crc_rx == crc_act_b)
        }
