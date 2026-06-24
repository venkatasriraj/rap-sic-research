import random
import numpy as np

from CHANNEL import SlowFadingChannel

class Simulation:

    def __init__(self, tx, rx, ch, chEst, slots=20, users=20, degree=2, K=32, Q=7):
        self.tx = tx
        self.rx = rx
        self.ch = ch
        self.chEst = chEst
        self.slots = slots
        self.degree = degree
        self.users = users
        self.K = K
        self.Q = Q
        self.userSlots = self.userSlotGen()

    def userSlotGen(self):
        userSlots = {}
        for i in range(1, self.slots+1):
            userSlots[i] = self.slotsGen(6*i+9)
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
        return [random.randint(0,1) for _ in range(self.K)]

    def frameBuild(self, FRAME):
        frame = {}
        key = FRAME.keys()
        h = self.ch.conRayleigh(self.users)
        for m in range(1, self.slots+1):
            if m not in key:
                frame[m] = self.ch.awgn_noise(self.K+1) 
            else:
                slotUsers = FRAME[m]
                signal = self.ch.awgn_noise(self.K+1)
                for u in slotUsers:
                    msg = self.msgGen(u)
                    sig_tx = self.tx.coeffCon(msg)
                    sig_power = np.mean( np.abs(sig_tx)**2 )
                    sig_tx /= sig_power
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
                msg_rx = self.rx.fftDizet(frame[slot], self.Q)
                # ---  MOCZ doesn't have any error detection mechanism in this simulation
                # Coefficients reconstruction using the message decoded
                sig_recon = self.tx.coeffCon(msg_rx)
                sig_power = np.mean( np.abs(sig_recon)**2 )
                sig_recon /= sig_power
                # h_est
                h_est = self.chEst.leastSquares(frame[slot], sig_recon)
                h_hat[slot] = h_est
                userId = bapm[slot][0]
                userSlots = self.userSlots[userId]
                for s in userSlots:
                    # print(f"@{slot} - userslots: {userSlots}")
                    if s != slot and s in interferencedSlots:
                        frame[s] -= sig_recon * h_est
                        pktsInSlots[s] -= 1
                        if pktsInSlots[s] < 1:
                            interferencedSlots.remove(s)
                            # print(f"Discarded Slot: {s}")
                # interferencedSlots.discard(slot)
                msg_hat[userId] = msg_rx
                # print(f"Interenced Slots: {interferencedSlots} - @{slot}")
        return dict(sorted(msg_hat.items(), reverse=False)), dict(sorted(h_hat.items(), reverse=False))

    def per(self, msg_hat):
        pcr, bcr = 0, 0
        for key, val in msg_hat.items():
            if np.all( val == self.msgGen(key) ):
                pcr += 1
            bcr += np.sum(val == self.msgGen(key))
        return pcr, bcr