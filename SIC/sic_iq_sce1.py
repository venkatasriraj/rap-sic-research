"""
Here we will be generating a packet for given user, modulate it and receive i-q samples
as output. Using those i-q samples we will decode the packet samples and get the transmitted bits.

For slot-2 we will add i-q samples of both users and using the decoded message from slot-1 
we will be reconstructing the i-q samples and remove the interference from slot-2 
to recover a packet.
"""

import numpy as np
import binascii
import csv
import time
import random

class irsa_pkt_gen():

    default_lambda = [
        (2, 0.50),
        (3, 0.78),
        (8, 1.00),
    ]

    access_code =[0xE1, 0x5A, 0xE8, 0x93]

    filler = [0xDE, 0xAD, 0xBE, 0xEE, 0xDE, 0xAD, 0xBE, 0xEE]

    max_degree = 16

    # def __init__():
