"""
Implementation of BPSK using designed classes for study purpose.
"""

import numpy as np

from packet import PacketStructure
from bpskTx import BPSKTransmitter
from bpskRx import BPSKReceiver

accessCode = [0xAA, 0xAA, 0xAA, 0xAA]
pktStr = PacketStructure(maxDegree=2, AccessCode=accessCode, pktSize=12)
tx = BPSKTransmitter()
rx = BPSKReceiver()

pktU1 = pktStr.buildPacket(userId=1, frameNo=1, degree=2, thisSlot=1, slotList=[1])
sigU1 = tx.modulate(pktU1)

pkt_hat = rx.demodulate(sigU1)
ber = rx.ber(pkt_hat, pktU1)