import random
import numpy as np

from CHANNEL import SlowFadingChannel

class Simulation:

    def __init__(self, tx, ch, chEst, slots=20, users=20, degree=2, pktSize=32, pilot=[1,0,1,0,1,0,1,0]):
        self.tx = tx
        # self.rx = rx
        self.ch = ch
        self.chEst = chEst
        self.slots = slots
        self.degree = degree
        self.users = users
        self.pktSize = pktSize
        self.pilot = pilot
        self.userSlots = self.userSlotGen()

    def userSlotGen(self):
        userSlots = {}
        for i in range(1, self.slots+1):
            userSlots[i] = self.slotsGen(i)
        return dict(sorted(userSlots.items(), reverse=False))

    def genBAPM(self, activeUsers):
        bapm = {}
        SLOT = set()
        for i in activeUsers:
            slots = self.userSlots[i]
            for s in slots:
                if s in SLOT:
                    bapm[s] += [i]
                else:
                    bapm[s] = [i]
                    SLOT.add(s)
        return dict(sorted(bapm.items(), reverse=False))            

    def slotsGen(self, i):
        random.seed(i)
        return sorted( random.sample(range(1, self.slots+1), self.degree) )

    def msgGen(self, i): # i is userId used as seed
        random.seed(i)
        return [random.randint(0,1) for _ in range(self.pktSize - len(self.pilot))]

    def frameBuild(self, FRAME):
        frame = {}
        key = FRAME.keys()
        h = self.ch.conRayleigh(self.users)
        for m in range(1, self.slots+1):
            if m not in key:
                frame[m] = self.ch.awgn_noise(self.pktSize+1) # DBPSK
            else:
                slotUsers = FRAME[m]
                signal = self.ch.awgn_noise(self.pktSize+1)
                for u in slotUsers:
                    msg = self.msgGen(u) 
                    pkt = np.append(self.pilot, msg).astype(np.int32)
                    sig_tx = self.tx.modulate(pkt)
                    # print(f"Signal Power: {np.mean(np.abs(sig_tx)**2)}")
                    signal += h[u-1] * sig_tx
                frame[m] = signal               
        return frame, h

    def frameParse(self, frame, bapm):
        pktsInSlots = {k:len(v) for k, v in bapm.items()}
        pktsInSlots = dict( sorted( pktsInSlots.items(), key=lambda item: item[1] ) )
        interferencedSlots = list(pktsInSlots.keys())
        msg_hat = {}
        h_hat = {}
        maxIter = self.users
        iterNo = 0
        # print(f"Interferenced Slots: {interferencedSlots}, Packets In Slots:{pktsInSlots}")
        while len(interferencedSlots) > 0 and iterNo < maxIter:
            if 1 not in pktsInSlots.values():        
                return dict(sorted(msg_hat.items(), reverse=False)), dict(sorted(h_hat.items(), reverse=False))
            slot = interferencedSlots[0]
            interferencedSlots.remove(slot)
            if pktsInSlots[slot] > 1:
                interferencedSlots += [slot]
            else:
                iterNo += 1
                pktsInSlots[slot] -= 1
                pkt_rx = self.tx.demodulate(frame[slot])
                # msg_rx = pkt_rx[len(self.pilot):]
                # ---  DBPSK doesn't have any error detection mechanism in this simulation
                # Coefficients reconstruction using the message decoded
                sig_recon = self.tx.modulate(pkt_rx)
                # h_est
                if len(self.pilot) == 0:
                    h_est = self.chEst.leastSquares(frame[slot], sig_recon)
                else:
                    h_est = self.chEst.leastSquares(frame[slot][:len(self.pilot)], sig_recon[:len(self.pilot)])
                userId = bapm[slot][0]  # it has only one element in the slot
                h_hat[userId] = h_est

                # -- we will be removing the userid after identifying the user since if ithas multiple 
                # elements it will always ouputs the same elements and it should also be removed from 
                # from other slots

                userSlots = self.userSlots[userId]
                for s in userSlots:
                    bapm[s].remove(userId)
                    # print(f"@{slot} - userslots: {userSlots}")
                    if s != slot and s in interferencedSlots:
                        frame[s] -= sig_recon * h_est
                        pktsInSlots[s] -= 1
                        if pktsInSlots[s] < 1:
                            interferencedSlots.remove(s)
                            # print(f"Discarded Slot: {s}")
                # interferencedSlots.discard(slot)
                msg_hat[userId] = pkt_rx
                # print(f"Interenced Slots: {interferencedSlots} - @{slot}")
        return dict(sorted(msg_hat.items(), reverse=False)), dict(sorted(h_hat.items(), reverse=False))

    def per(self, pkt_hat):
        pcr, bcr = 0, 0
        for key, val in pkt_hat.items():
            msg = self.msgGen(key)
            pkt = np.append(self.pilot, msg)
            if np.all( val ==  pkt):
                pcr += 1
            bcr += np.sum(val == pkt)
        return pcr, bcr

    @staticmethod
    def mae(h, h_hat, uId):
        if uId in h_hat.keys():
            return np.abs(h[uId-1] - h_hat[uId]), 1
        else:
            return 0, 0