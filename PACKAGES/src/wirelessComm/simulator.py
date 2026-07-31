def simulator(sim, load, noIter, sicMode = True, uId = 1):

    PER, BER, THROUGHPUT, MAE, MAE_count = 0, 0, 0, 0, 1e-10
    for i in range(noIter):
        userSlotsGen = sim.userSlotGen()
        FRAME = {}
        slot = set()
        activeUsers = sorted( sim.rng.sample( range(1, sim.users+1), int(load*sim.users) ) )

        for userId in activeUsers:
            userSlot = userSlotsGen[userId]
            for s in userSlot:
                if s not in slot:
                    FRAME[s] = [userId]
                    slot.add(s)
                else:
                    FRAME[s] += [userId]
        FRAME = dict(sorted( FRAME.items(), reverse=False ))
        frame, h = sim.frameBuild(FRAME)
        frameBAPM = sim.genBAPM(activeUsers, userSlotsGen)
        if sicMode:
            pkt_hat, h_hat = sim.frameParse(frame, frameBAPM, userSlotsGen)
            if uId in activeUsers:
                mae_temp, count = sim.maeh(h, h_hat, uId)
                MAE += mae_temp
                MAE_count += count
        else:
            pkt_hat = sim.frameParseNoSIC(frame, frameBAPM, userSlotsGen)
        pcr, bcr_frame = sim.per(pkt_hat)
        PER += ( 1 - (pcr/len(activeUsers)) )
        BER += ( 1 - (bcr_frame / ( sim.pktSize * len(activeUsers) )) )
        THROUGHPUT += pcr / len(activeUsers)
    if sicMode:
        return PER, BER, THROUGHPUT, MAE, MAE_count
    else:
        return PER, BER, THROUGHPUT