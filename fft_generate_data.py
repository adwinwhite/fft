#! /usr/bin/python

import sys
import struct
import math

with open(sys.argv[1], "wb") as outputfile:
    for i in range(int(sys.argv[2])):
        outputfile.write(struct.pack('d', math.sin(i * 0.1)))
        outputfile.write(struct.pack('d', 0))
