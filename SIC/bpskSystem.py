"""
Design of BPSK system.
"""

import numpy as np
from BPSK import (
    BPSKBase,
    IRSAReceiver,
    IRSATransmitter
)
from CHANNEL import (
    SlowFadingChannel,
    ChannelEstimation
)

# class BPSKSystem:

    