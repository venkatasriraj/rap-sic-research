import random
import numpy as np

class dbpskSIMULATION:

    def __init__(self, tx, ch, chEst, slots=20, users=20, degree=2, 
                pktSize=31, pilot=[1,0,1,0,1,0,1,0], seed = None, maxIter='None'):
        self.tx = tx
        self.ch = ch
        self.chEst = chEst
        self.slots = slots
        self.degree = degree
        self.users = users
        self.pktSize = pktSize
        self.pilot = pilot
        self.maxIter = users * 2 if maxIter == 'None' else maxIter
        self.rng = random.Random(seed)

    def userSlotGen(self):
        userSlots = {}
        for i in range(1, self.users+1):
            userSlots[i] = self.rng.sample(range(1, self.slots+1), self.degree)
        return dict(sorted(userSlots.items(), reverse=False))         

    def msgGen(self, i): 
# i is userId used as seed - to reproduce the msg at Rx for performance evaluation
        rng = random.Random(i)
        return [rng.randint(0,1) for _ in range(self.pktSize - len(self.pilot))]

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
                    pkt = np.append(self.pilot, msg).astype(np.int32)
                    sig_tx = self.tx.differentialModulation(pkt)
                    signal += h[u-1] * sig_tx
            frame[m] = signal    
        return frame, h

    def frameParse(self, frame, bapm, userSlots):
# pktsInSlots - in which slots the packets are transmitted and sorts them based on packets transmitted in that slot
        pktsInSlots = {k:len(v) for k, v in bapm.items()}
        pktsInSlots = dict( sorted( pktsInSlots.items(), key=lambda item: item[1] ) )
        interferencedSlots = list(pktsInSlots.keys())
        msg_hat = {}
        h_hat = {}
        iterNo = 0
        while len(interferencedSlots) > 0 and iterNo < self.maxIter:
# this is used to reduce the simulation load by checking if there are any "SINGLETON" slots
            if 1 not in pktsInSlots.values():        
                return dict(sorted(msg_hat.items(), reverse=False)), dict(sorted(h_hat.items(), reverse=False))
            slot = interferencedSlots[0]
# the slot is poped and the added if there are more than 1 packet in a slot
            interferencedSlots.remove(slot)
# this evaluation plays a role when we deactivate the above "SINGLETON" condition 
            if pktsInSlots[slot] == 1:
                iterNo += 1
# it has only one element in the slot - since we are only evaluating "SINGLETON" slots
# if not: we need to identify the userId from the "PACKET HEADER"
                userId = bapm[slot][0] 
# we will be removing the userid after identifying the user since if it has multiple elements 
# it will always ouputs the same elements and it should also be removed from other slots
                uSlot = userSlots[userId]
                pktsInSlots[slot] -= 1
                pktRx = self.tx.differentialDemodulation(frame[slot])
# Coefficients reconstruction using the message decoded
                sig_recon = self.tx.differentialModulation(pktRx)
                h_est = self.chEst.leastSquares(frame[slot][:len(self.pilot) or None], sig_recon[:len(self.pilot) or None])
                # if len(self.pilot) == 0:
                #     h_est = self.chEst.leastSquares(frame[slot], sig_recon)
                # else:
                #     h_est = self.chEst.leastSquares(frame[slot][:len(self.pilot)], sig_recon[:len(self.pilot)])
                h_hat[userId] = h_est

                for s in uSlot:
                    bapm[s].remove(userId)
                    if s != slot and s in interferencedSlots:
                        frame[s] -= sig_recon * h_est
                        pktsInSlots[s] -= 1
                        if pktsInSlots[s] < 1:
                            interferencedSlots.remove(s)
                msg_hat[userId] = pktRx
            else:
                interferencedSlots += [slot]                
        return dict(sorted(msg_hat.items(), reverse=False)), dict(sorted(h_hat.items(), reverse=False))

    def frameParseNoSIC(self, frame, bapm):
        pktsInSlots = {k:len(v) for k, v in bapm.items()}
        pktsInSlots = dict( sorted( pktsInSlots.items(), key=lambda item: item[1] ) )
        interferencedSlots = list(pktsInSlots.keys())
        msg_hat = {}
# checks the "SINGLETON" slots in the frame
        while 1 in pktsInSlots.values():
            slot = interferencedSlots[0]
            interferencedSlots.remove(slot)
            pktsInSlots[slot] -= 1
            msg_rx = self.tx.differentialDemodulation(frame[slot])
            userId = bapm[slot][0]
            msg_hat[userId] = msg_rx
        return dict(sorted(msg_hat.items(), reverse=False))

    # perfect CSI is available at Rx
    def frameParseCSI(self, frame, bapm, userSlots, h):
        pktsInSlots = {k:len(v) for k, v in bapm.items()}
        pktsInSlots = dict(sorted(pktsInSlots.items(), key= lambda item: item[1]))
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
                uSlots = userSlots[userId]
                pktsInSlots[slot] -= 1
                msg_rx = self.tx.differentialDemodulation(frame[slot])
                msg_hat[userId] = msg_rx 
                sig_recon = self.tx.differentialModulation(msg_rx)
                for s in uSlots:
                    bapm[s].remove(userId)
                    if s != slot and s in interferencedSlots:
                        frame[s] -= sig_recon * h[userId-1]
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