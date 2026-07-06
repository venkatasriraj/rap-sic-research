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
                    sig_tx /= np.sqrt(sig_power)
                    signal += h[u-1] * sig_tx
                frame[m] = signal               
        return frame, h

    def frameParse(self, frame, bapm):
        pktsInSlots = {k:len(v) for k, v in bapm.items()}
        pktsInSlots = dict( sorted( pktsInSlots.items(), key=lambda item: item[1] ) )
        interferencedSlots = list(pktsInSlots.keys())
        msg_hat = {}
        h_hat = {}
        iterNo = 0
        # singleton = []
        # users_decoded = []
        while len(interferencedSlots) > 0 and iterNo < self.users:
            if 1 not in pktsInSlots.values():        
                return dict(sorted(msg_hat.items(), reverse=False)), dict(sorted(h_hat.items(), reverse=False))
            slot = interferencedSlots[0]
            interferencedSlots.remove(slot)
            if pktsInSlots[slot] > 1:
                interferencedSlots += [slot]
            else:
                # singleton += [slot]
                iterNo += 1
                pktsInSlots[slot] -= 1
                msg_rx = self.rx.fftDizet(frame[slot], self.Q)
                # ---  MOCZ doesn't have any error detection mechanism in this simulation
                # Coefficients reconstruction using the message decoded
                sig_recon = self.tx.coeffCon(msg_rx)
                sig_power = np.mean( np.abs(sig_recon)**2 )
                sig_recon /= np.sqrt(sig_power)
                # h_est
                h_est = self.chEst.leastSquares(frame[slot], sig_recon)
                # h_est = self.chEst.modifiedLS(frame[slot], sig_recon)
                userId = bapm[slot][0]
                h_hat[userId] = h_est
                # if userId == 1:
                #     print(f"    Estimated h is {h_est}")
                userSlots = self.userSlots[userId]
                # users_decoded += [userId]
                for s in userSlots:
                    bapm[s].remove(userId)
                    if s != slot and s in interferencedSlots:
                        frame[s] -= sig_recon * h_est
                        pktsInSlots[s] -= 1
                        if pktsInSlots[s] < 1:
                            interferencedSlots.remove(s)
                msg_hat[userId] = msg_rx
        # print(f"    Slots decoded for a frame: {singleton}, {len(singleton)}")
        # print(f"    Users Decoded for a frame: {users_decoded}, {len(users_decoded)}")
        return dict(sorted(msg_hat.items(), reverse=False)), dict(sorted(h_hat.items(), reverse=False))

    def per(self, msg_hat):
        pcr, bcr = 0, 0
        for key, val in msg_hat.items():
            if np.all( val == self.msgGen(key) ):
                pcr += 1
            bcr += np.sum(val == self.msgGen(key))
        return pcr, bcr

    @staticmethod
    def maeh(h, h_hat, userId):
        # print(f"h: {h}\n")
        # print(f"UserId: {userId}")
        # print(f"hEst: {h_hat}")
        if userId in h_hat.keys():
            # print(f"h: {h[userId-1]}, h_hat: {h_hat[userId]}, MAE is {abs(h[userId-1]-h_hat[userId])}")
            return abs(h[userId-1]-h_hat[userId]), 1
        else:
            # print(f"UserId not available in this frame")
            return 0, 0