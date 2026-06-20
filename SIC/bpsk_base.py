"""
Implementation of BPSK using designed classes for study purpose.
Scenario: We have considered a single frame with 2 slots.
Where user 1 transmits on slot-1 and user 2 transmits on slot-2.

Since we are using Differnetial Coding scheme the slot length will be increased by 1B.
"""

import numpy as np

from BPSK import (
    PacketStructure,
    IRSAReceiver,
    IRSATransmitter
)
from CHANNEL import (
    SlowFadingChannel,
    ChannelEstimation
)

SNR_dB = 20

# accessCode = [0xAA, 0xAA, 0xAA, 0xAA]
accessCode = [0xE1, 0x5A, 0xE8, 0x93]
# accessCode = [0xBB, 0xAA, 0xAA, 0xBB]
pktSize = 16 # with CRC(4B)
maxDegree =2
slotLen = 8 * (pktSize + 1)

pktStr = PacketStructure(maxDegree=maxDegree, AccessCode=accessCode, pktSize=pktSize)
tx = IRSATransmitter(slotLen=slotLen)
rx = IRSAReceiver(accessCode)
chEst = ChannelEstimation()

pktU1 = pktStr.buildPacket(userId=1, frameNo=1, degree=1, thisSlot=1, slotList=[1])
sigU1 = tx.pkt_to_iq(pktU1)
sigU1_power = np.mean(np.abs(sigU1))
# print(f"Calculated signal power is {sigU1_power}")

noise_var = sigU1_power * 10**(-SNR_dB/10)
ch = SlowFadingChannel(noise_var)
sigS1, chCoeffU1 = ch.transmit(sigU1)
print(f"Channel Coefficient for User-1 {chCoeffU1}")
# pkt_hat = rx.demodulate(sigU1)

pkt_hat, acReceived = rx.decodeSlot(sigS1)
pkt_info = pktStr.parsePacket(pkt_hat)
# CRC is not passed

if pkt_info and pkt_info["crc_ok"] == True:
    print(f"packet decoded correctly.")

#  -- here we will be implementing packet reconstruction to check whether
#   the transmitted and received packet are same
pktU1_recon = pktStr.buildPacket(
    userId=pkt_info["userId"], frameNo=pkt_info["frameNo"],
    degree=pkt_info["degree"], thisSlot=pkt_info["thisSlot"],
    slotList=pkt_info["slotList"]
)

sigU1_recon = tx.pkt_to_iq(pktU1_recon)
if np.all(sigU1 == sigU1_recon):
    print("Successfully reconstructed the packet received.")

acBits = tx.bytes_to_bits(accessCode)
acModulated = tx.modulate(acBits)

print(f"acReceiverd: {acReceived}")
chEstU1 = chEst.leastSquares(acReceived, acModulated)
print(f"Channel Coefficient: {chCoeffU1}, Estimated Channel Coefficient: {chEstU1}")

ber = rx.ber(pkt_hat, pktU1)