"""
DBPSK + IRSA. Access code will be used as pilot sequence to estimate channel.
PER and SIC analysis (over reconstruction and recovery) over SNR.
[AC(4) | U(1) | F(1) | D(1) | tS(1) | SL(2) | Data(pkt-14) | CRC(4)]
For BPSK Signal Power = 1.

In IRSA we should be sending the same packet in a given frame. But we are modified the 
packetStructure and added thisSlot attribute to identify the position of the packet 
in the slot and perform interference cancellation.
"""
import numpy as np

from CHANNEL import (
    SlowFadingChannel,
    ChannelEstimation
)
from BPSK import (
    IRSATransmitter,
    IRSAReceiver,
    PacketStructure
)

accessCode = [0xBB, 0xAA, 0xAA, 0xBB]
maxDegree = 2
pktSize = 32 # including CRC 
slotLen = (pktSize + 1) * 8

pktStr = PacketStructure(maxDegree=maxDegree, AccessCode=accessCode, pktSize=pktSize)
tx = IRSATransmitter(slotLen=slotLen)
rx = IRSAReceiver(accessCode=accessCode)
chEst = ChannelEstimation()

SNR_dB = np.arange(-10, 21, 2)
noIter = 1000

per = {}; mae = {}; throughput = {}
for snr in SNR_dB:
    PCR = [0, 0]
    MAE = 0; mae_count = 1e-10; 
    noise_var = 10**(-snr / 10)
    ch = SlowFadingChannel(var=noise_var)
    for i in range(noIter):
        pktU11 = pktStr.buildPacket(1, 1, 2, 1, [1,2])
        sigU11 = tx.pkt_to_iq(pktU11)
        pktU12 = pktStr.buildPacket(1,1,2,2,[1,2])
        sigU12 = tx.pkt_to_iq(pktU12)
        chCoeffU1 = np.sqrt(1/2) * ( np.random.randn() + 1j*np.random.randn() )

        pktU22 = pktStr.buildPacket(2,1,1,2,[2])
        sigU22 = tx.pkt_to_iq(pktU22)
        chCoeffU2 = np.sqrt(1/2) * ( np.random.randn() + 1j*np.random.randn() )

        #  --- Slot-1: U1 through channel, Slot-2: U1 + U2 @channel
        sigS1 = ch.sic_transmit([sigU11], [chCoeffU1])
        sigS2 = ch.sic_transmit([sigU12, sigU12], [chCoeffU1, chCoeffU2])

        # -- packet decoding from Slot1 and recon to perform SIC
        pktU1_hat, acRecU1 = rx.decodeSlot(sigS1)
        if pktU1_hat is None:
            continue
        pktU1_info = pktStr.parsePacket(pktU1_hat)
        if pktU1_info and pktU1_info["crc_ok"]:
            PCR[0] += 1

            thisSlot = [i for i in pktU1_info["slotList"] if i != pktU1_info["thisSlot"]]
            pktU1_recon = pktStr.buildPacket(
                userId=pktU1_info["userId"], frameNo=pktU1_info["frameNo"],
                degree=pktU1_info["frameNo"], thisSlot=thisSlot[0],
                slotList=pktU1_info["slotList"]
            )
            sigS1_recon = tx.pkt_to_iq(pktU1_recon)

            #  --  Channel Estimation using the received and recon AC
            acBits = tx.bytes_to_bits(accessCode)
            acModulated = tx.modulate(acBits)

            chEstU1 = chEst.leastSquares(acRecU1, acModulated)
            
            #  ---  SIC implementation
            sigS2_can = sigS2 - chEstU1 * sigS1_recon
            pktU2_hat, _ = rx.decodeSlot(sigS2_can)
            if pktU2_hat is None:
                continue
            pktU2_info = pktStr.parsePacket(pktU2_hat)
            if pktU2_info and pktU2_info["crc_ok"]:
                PCR[1] += 1

            # -- MSE of channel coefficient estimate
            MAE += np.abs(chEstU1 - chCoeffU1)
            mae_count += 1
    per[snr] = 1 - ( np.array(PCR) / noIter )
    mae[snr] = MAE / mae_count
    throughput[snr] = np.array(PCR) / noIter

# for snr, per in PER.items():
#     print(f"SNR(dB): {snr}, PER: {per}, MSE of h: {MSE[snr]}")
plt.figure(1, dpi=800)
y = PER.values()
for i in range(2):
    plt.plot(PER.keys(), y[i], linestyle='-', linewidth=0.9, label=f"Slot-{i+1}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("SNR(dB)")
plt.ylabel("Packet Error Rate")
plt.title(f"PER vs SNR over {noIter} frames")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("results/perTest1.jpeg")

plt.figure(2, dpi=800)
plt.plot(mae.keys(), mae.values(), linestyle='-', linewidth=0.9)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("SNR(dB)")
plt.ylabel("MAE of h_est")
plt.title(f"MAE of h_est vs SNR over {noIter} frames")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("results/maehTest1.jpeg")

plt.figure(3, dpi=800)
y = throughput.values()
for i in range(2):
    plt.plot(throughput.keys(), y[i], linestyle='-', linewidth=0.9, label=f"Slot-{i+1}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("SNR(dB)")
plt.ylabel("Throughput (t)")
plt.title(f"Throughput vs SNR over {noIter} frames")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("results/thrTest1.jpeg")