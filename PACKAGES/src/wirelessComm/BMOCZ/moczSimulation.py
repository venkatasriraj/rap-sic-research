import random
import numpy as np

class moczSIMULATION:

    def __init__(self, system, ch, chEst, slots=20, users=20, degree=2, 
                 packetSize=31, Q=32, seed=None, maxIter='None', energy='None'):
        self.system = system
        self.ch = ch
        self.chEst = chEst
        self.slots = slots
        self.degree = degree
        self.users = users
        self.pktSize = packetSize
        self.Q = Q
        self.maxIter = users * 2 if maxIter == 'None' else maxIter
        self.energy = packetSize + 1 if energy == 'None' else energy
        self.rng = random.Random(seed)

    def userSlotGen(self):
        userSlots = {}
        for i in range(1, self.users+1):
            userSlots[i] = self.rng.sample(range(1, self.slots+1), self.degree)
        return dict(sorted(userSlots.items(), reverse=False))           

    def msgGen(self, i): # i is userId used as seed
        rng = random.Random(i)
        return [rng.randint(0,1) for _ in range(self.pktSize)]

    def frameBuild(self, FRAME):
        frame = {}
        key = FRAME.keys()
        h = self.ch.conRayleigh(self.users)
        for m in range(1, self.slots+1):
            signal = self.ch.awgn_noise(self.pktSize+1)
            if m in key:
                slotUsers = FRAME[m]
                for u in slotUsers:
                    msg = self.msgGen(u)
                    sigTx = self.system.coeffCon(msg)
                    l2_norm = np.linalg.norm(sigTx, ord=2) 
                    sigTx = (sigTx / l2_norm) * np.sqrt(self.energy) 
                    signal += h[u-1] * sigTx
            frame[m] = signal              
        return frame, h

    def frameParse(self, frame, bapm, userSlots):
        pktsInSlots = {k:len(v) for k, v in bapm.items()}
        pktsInSlots = dict( sorted( pktsInSlots.items(), key=lambda item: item[1] ) )
        interferencedSlots = list(pktsInSlots.keys())
        msg_hat = {}
        h_hat = {}
        iterNo = 0
        while len(interferencedSlots) > 0 and iterNo < self.maxIter:
            if 1 not in pktsInSlots.values():        
                return dict(sorted(msg_hat.items(), reverse=False)), dict(sorted(h_hat.items(), reverse=False))
            slot = interferencedSlots[0]
            interferencedSlots.remove(slot)
            if pktsInSlots[slot] == 1:
                iterNo += 1
                userId = bapm[slot][0]
                uSlot = userSlots[userId]
                pktsInSlots[slot] -= 1
                msgRx = self.system.fftDizet(frame[slot], self.Q)
# Coefficients reconstruction using the message decoded
                sigRecon = self.system.coeffCon(msgRx)
                l2_norm = np.linalg.norm(sigRecon, ord=2) 
                sigRecon = (sigRecon / l2_norm) * np.sqrt(self.energy)
                
                h_est = self.chEst.leastSquares(frame[slot], sigRecon)
                h_hat[userId] = h_est

                for s in uSlot:
                    bapm[s].remove(userId)
                    if s != slot and s in interferencedSlots:
                        frame[s] -= sigRecon * h_est
                        pktsInSlots[s] -= 1
                        if pktsInSlots[s] < 1:
                            interferencedSlots.remove(s)
                msg_hat[userId] = msgRx
            else:
                interferencedSlots += [slot]
        return dict(sorted(msg_hat.items(), reverse=False)), dict(sorted(h_hat.items(), reverse=False))

    def frameParseNoSIC(self, frame, bapm):
        pktsInSlots = {k:len(v) for k, v in bapm.items()}
        pktsInSlots = dict( sorted( pktsInSlots.items(), key=lambda item: item[1] ) )
        interferencedSlots = list(pktsInSlots.keys())
        msg_hat = {}
        while 1 in pktsInSlots.values():
            slot = interferencedSlots[0]
            interferencedSlots.remove(slot)
            pktsInSlots[slot] -= 1
            msg_rx = self.system.fftDizet(frame[slot], self.Q)
            userId = bapm[slot][0]
            msg_hat[userId] = msg_rx
        return dict(sorted(msg_hat.items(), reverse=False))

    # perfect channel state information (CSI) is available at Rx
    def frameParseCSI(self, frame, bapm, userSlots, h):
        pktsInSlots = {k:len(v) for k, v in bapm.items()}
        pktsInSlots = dict(sorted(pktsInSlots.items(), key = lambda item: item[1]))
        interferencedSlots = list(pktsInSlots.keys())
        msg_hat = {}
        iterNo = 0
        while len(interferencedSlots) > 0 and iterNo < self.maxIter:
            if 1 not in pktsInSlots.values():
                return dict(sorted(msg_hat.items(), reverse=False))
            slot = interferencedSlots[0]
            interferencedSlots.remove(slot)
            if pktsInSlots[slot] == 1:
                iterNo += 1
                userId = bapm[slot][0]
                uSlot = userSlots[userId]
                pktsInSlots[slot] -= 1
                msgRx = self.system.fftDizet(frame[slot], self.Q)
                msg_hat[userId] = msgRx
                sigRecon = self.system.coeffCon(msgRx)
                l2_norm = np.linalg.norm(sigRecon, ord=2) 
                sigRecon = (sigRecon / l2_norm) * np.sqrt(self.energy)
                
                for s in uSlot:
                    bapm[s].remove(userId)
                    if s != slot and s in interferencedSlots:
                        frame[s] -= sigRecon * h[userId-1]
                        pktsInSlots[s] -= 1
                        if pktsInSlots[s] < 1:
                            interferencedSlots.remove(s)
            else:
                interferencedSlots += [slot]
        return dict(sorted(msg_hat.items(), reverse=False))

    def pcr(self, msg_hat):
        pcr, bcr = 0, 0
        for key, val in msg_hat.items():
            if np.all( val == self.msgGen(key) ):
                pcr += 1
            bcr += np.sum(val == self.msgGen(key))
        return pcr, bcr

    @staticmethod
    def maeh(h, h_hat, userId):
        if userId in h_hat.keys():
            return abs(h[userId-1] - h_hat[userId])/abs(h[userId-1]), 1
        else:
            return 0, 0