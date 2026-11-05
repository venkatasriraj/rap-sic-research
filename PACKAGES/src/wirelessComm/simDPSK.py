"""
Initially we considered block-fading channel(quasi-static) without CFO
stage-2: Consider CFO for block-fading channel
stage-3: consider frequency selective channel without CFO
stage-4: consider frequency selective channel with CFO

Rotation affects on DPSK and MOCZ vary. For MOCZ rotation PHI causes the zeroes to rotate in
anti-clockwise direction. Depending on the scheme the method to reconstruct the signal varies. 
"""
import numpy as np

class SIMULATORDPSK:

    def __init__(self, system, ch, chEst, slots=20, users=20, degree=2, pktSize=31, 
                 Q=32, maxIter='None', energy='None', scheme='DPSK', chMode='BF', cfo=False):
        self.system = system
        self.ch = ch
        self.chEst = chEst
        self.slots = slots
        self.users = users
        self.degree = degree
        self.pktSize = pktSize
        self.Q = Q
        self.maxIter = users * 2 if maxIter == 'None' else maxIter
        self.energy = pktSize + 1 if energy == 'None' else energy
        self.scheme = scheme
        self.chMode = chMode
        self.CFO = cfo
        # self.pilot = [1,0,1,0,1,0,1,0] if pilot == 'None' else pilot
        self.rng = np.random.default_rng()

# BAPM - binary access pattern matrix
# it contains m slots and details of users transmitted in a slot
    def genBAPM(self, activeUsers, userSlots):
        bapm, msg = {}, {}
# slot-set is used to create a list for a given slot by checking whether it is initialised or not
        SLOT = set()
        for i in activeUsers:
            msg[i] = np.random.randint(0, 2, self.pktSize)
            slots = userSlots[i]
            for s in slots:
                if s in SLOT:
                    bapm[s] += [i]
                else:
                    bapm[s] = [i]
                    SLOT.add(s)
        # return dict(sorted(bapm.items(), reverse=False)), dict(sorted(msg.items(), reverse=False))  
        return bapm, msg

    def userSlotGen(self):
        userSlots = {}
        for i in range(self.users):
            userSlots[i] = self.rng.choice(self.slots, size=self.degree)
        # return dict(sorted(userSlots.items(), reverse=False)) 
        return userSlots

    def frameBuild(self, frameBAPM, msg):
        frame = {}
        key = frameBAPM.keys()
        h = self.ch.conRayleigh(self.users)
        rotation = np.random.uniform(0, 2*np.pi)
        cfoCount = 0
# we will deactivate "cfoCount", for constant CFO it doesn't have an impact
        for m in range(self.slots):
            signal = self.ch.awgn_noise(self.pktSize+1)
            if m in key:
                slotUsers = frameBAPM[m]
                for u in slotUsers:
                    signal += self.system.transmitterRAP(msg[u], self.energy) * h[u]
# we are considering constant-rotation CFO impairment at the receiver
            if self.CFO:
                sigRx = self.ch.CFO(signal, rotation, noiseFlag=False, cfoCount=0) #  cfoCount=cfoCount
                cfoCount += len(sigRx)
            frame[m] = signal 
        return frame, h 

    def frameParse(self, frame, frameBAPM, userSlotsGen):
# pktsInSlots - in which slots the packets are transmitted and sorts them based on packets transmitted in that slot
        pktsInSlots = {k:len(v) for k, v in frameBAPM.items()}
        pktsInSlots = dict( sorted( pktsInSlots.items(), key=lambda item: item[1] ) )
        interferencedSlots = list(pktsInSlots.keys())
        msgRx, h_hat, iterNo = {}, {}, 0
        while len(interferencedSlots) > 0 and iterNo < self.maxIter:
# this is used to reduce the simulation load by checking if there are any "SINGLETON" slots
            if 1 not in pktsInSlots.values():        
                return dict(sorted(msgRx.items(), reverse=False)), dict(sorted(h_hat.items(), reverse=False))
            slot = interferencedSlots[0]
# the slot is poped and the added if there are more than 1 packet in a slot
            interferencedSlots.remove(slot)
# this evaluation plays a role when we deactivate the above "SINGLETON" condition 
            if pktsInSlots[slot] == 1:
                iterNo += 1
# it has only one element in the slot - since we are only evaluating "SINGLETON" slots
# if not: we need to identify the userId from the "PACKET HEADER"
                userId = frameBAPM[slot][0] 
# we will be removing the userid after identifying the user since if it has multiple elements 
# it will always ouputs the same elements and it should also be removed from other slots
                uSlot = userSlotsGen[userId]
                pktsInSlots[slot] -= 1
                if self.scheme == 'DPSK':
                    pktRx, phiEst, sigRxCorrected = self.system.receiverRAP(frame[slot], self.CFO)
                else:
                    pktRx, phiEst, sigRxCorrected = self.system.receiverRAP(frame[slot], self.CFO, self.acpc) 
# CFO estimation for MOCZ schemes varies unlike DPSK. For MOCZ, zeroes rotate counter-clockwise direction,
# while we estimate the CFO in clock-wise direction. To estimate the CFO in counter-clockwise direction, we consider
                    phiEst = 2*np.pi - phiEst
# Coefficients reconstruction using the message decoded
                sigRecon = self.system.transmitterRAP(pktRx, self.energy)

                if self.CFO:
                    cfoCount = 0 # slot * (self.pktSize + 1)
# we use the CFO corrected signal and reconstructed signal to estimate the block-fading channel
                    h_est = self.chEst.leastSquares(sigRxCorrected, sigRecon)

                    sigReconSIC = self.ch.CFO(sigRecon * h_est, phiEst, noiseFlag=False, cfoCount=cfoCount)
                else:
                    h_est = self.chEst.leastSquares(frame[slot], sigRecon)
                    sigReconSIC = sigRecon * h_est
#----------------TO BE CONTINUED---------------------
                h_hat[userId] = h_est

                for s in uSlot:
                    frameBAPM[s].remove(userId)
                    if s != slot and s in interferencedSlots:
                        frame[s] -= sigReconSIC  # sigRecon * h_est
                        pktsInSlots[s] -= 1
                        if pktsInSlots[s] < 1:
                            interferencedSlots.remove(s)
                msgRx[userId] = pktRx
            else:
                interferencedSlots += [slot]                
        # return dict(sorted(msgRx.items(), reverse=False)), dict(sorted(h_hat.items(), reverse=False))
        return msgRx, h_hat
    

    def frameParseNoSIC(self, frame, frameBAPM):
        pktsInSlots = {k:len(v) for k, v in frameBAPM.items()}
        pktsInSlots = dict( sorted( pktsInSlots.items(), key=lambda item: item[1] ) )
        interferencedSlots = list(pktsInSlots.keys())
        msg_hat = {}
        while 1 in pktsInSlots.values():
            slot = interferencedSlots[0]
            interferencedSlots.remove(slot)
            pktsInSlots[slot] -= 1
            userId = frameBAPM[slot][0]
            if self.scheme == 'DPSK':
                msg_hat[userId], _, _ = self.system.receiverRAP(frame[slot], self.CFO)
            else:
                msg_hat[userId], _, _ = self.system.receiverRAP(frame[slot], self.CFO)
        # return dict(sorted(msg_hat.items(), reverse=False))
        return msg_hat

    def frameParseCSI(self, frame, frameBAPM, userSlotsGen, h):
        pktsInSlots = {k:len(v) for k, v in frameBAPM.items()}
        pktsInSlots = dict(sorted(pktsInSlots.items(), key = lambda item: item[1]))
        interferencedSlots = list(pktsInSlots.keys())
        msg_hat, iterNo = {}, 0
        while len(interferencedSlots) > 0 and iterNo < self.maxIter:
            if 1 not in pktsInSlots.values():
                return dict(sorted(msg_hat.items(), reverse=False))
            slot = interferencedSlots[0]
            interferencedSlots.remove(slot)
            if pktsInSlots[slot] == 1:
                iterNo += 1
                userId = frameBAPM[slot][0]
                uSlot = userSlotsGen[userId]
                pktsInSlots[slot] -= 1
                if self.scheme == 'DPSK':
                    msg_hat[userId], phiEst, sigRxCorrected = self.system.receiverRAP(frame[slot], self.CFO)
                else:
                    msg_hat[userId], phiEst, sigRxCorrected = self.system.receiverRAP(frame[slot], self.CFO)
                    phiEst = 2*np.pi - phiEst
                sigRecon = self.system.transmitterRAP(msg_hat[userId], self.energy)

                if self.CFO:
                    cfoCount = 0 # slot * (self.pktSize + 1)
                    h_est = self.chEst.leastSquares(sigRxCorrected, sigRecon)
                    sigReconSIC = self.ch.CFO(sigRecon * h_est, phiEst, noiseFlag=False, cfoCount=cfoCount)
                else:
                    sigReconSIC = sigRecon * h[userId]

                for s in uSlot:
                    frameBAPM[s].remove(userId)
                    if s != slot and s in interferencedSlots:
                        frame[s] -= sigReconSIC # sigRecon * h[userId] # h[userId-1]
                        pktsInSlots[s] -= 1
                        if pktsInSlots[s] < 1:
                            interferencedSlots.remove(s)
            else:
                interferencedSlots += [slot]
        # return dict(sorted(msg_hat.items(), reverse=False))
        return msg_hat
    
    def simulator(self, load, noIter, sicMode = 'SIC', uId = 1):

        PER, BER, THROUGHPUT, MAE, MAE_count = 0, 0, 0, 0, 1e-10
        for _ in range(noIter):
            userSlotsGen = self.userSlotGen()
            activeUsers = sorted( self.rng.choice(20, size=int(load*self.users), replace=False) )
            
            frameBAPM, msg = self.genBAPM(activeUsers, userSlotsGen)
            frame, h = self.frameBuild(frameBAPM, msg)
            
            if sicMode == "SIC":
                pkt_hat, h_hat = self.frameParse(frame, frameBAPM, userSlotsGen)
                if uId in activeUsers:
                    mae_temp, count = self.maeh(h, h_hat, uId)
                    MAE += mae_temp
                    MAE_count += count
            elif sicMode == "noSIC":
                pkt_hat = self.frameParseNoSIC(frame, frameBAPM)
            elif sicMode == "CSI":
                pkt_hat = self.frameParseCSI(frame, frameBAPM, userSlotsGen, h)
            pcr, bcr_frame = self.pcr(pkt_hat, msg)
            PER += ( 1 - (pcr/len(activeUsers)) )
            BER += ( 1 - (bcr_frame / ( self.pktSize * len(activeUsers) )) )
            THROUGHPUT += pcr / self.slots
        if sicMode == "SIC":
            return PER, BER, THROUGHPUT, MAE, MAE_count
        else:
            return PER, BER, THROUGHPUT

    def pcr(self, msg_hat, msg):
        pcr, bcr = 0, 0
        for key, val in msg_hat.items():
            # print(msg[key], val)
            if np.all( val == msg[key] ):
                pcr += 1
            bcr += np.sum(val == msg[key])
        return pcr, bcr

    @staticmethod
    def maeh(h, h_hat, userId):
        if userId in h_hat.keys():
            return abs(h[userId-1] - h_hat[userId])/abs(h[userId-1]), 1
        else:
            return 0, 0