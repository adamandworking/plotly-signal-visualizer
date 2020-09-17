import numpy as np
import serial
import time


class ReadLine:
    def __init__(self, s):
        self.buf = bytearray()
        self.s = s
    
    def readline(self):
        i = self.buf.find(b"\n")
        if i >= 0:
            r = self.buf[:i+1]
            self.buf = self.buf[i+1:]
            return r
        while True:
            i = max(1, min(2048, self.s.in_waiting))
            data = self.s.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r = self.buf + data[:i+1]
                self.buf[0:] = data[i+1:]
                return r
            else:
                self.buf.extend(data)

# Source code from https://github.com/pyserial/pyserial/issues/216#issuecomment-369414522

COM_PORT = 'COM5'
BAUD_RATES = 115200
ser = serial.Serial(COM_PORT, BAUD_RATES)
rl = ReadLine(ser)
count = 0
while True:
    raw_data = rl.readline()
    print("raw_data", raw_data)
    # should_print = str(count) + '\r\n'
    # if raw_data != should_print.encode():
    #     print('something wrong!!!')
    #     exit()
    count += 1
# ser.setDTR(False)
# ser.setRTS(False)
# count = 0
# # time.sleep(2)

# while 1:
#     # data = ser.read_until('a'.encode())
#     data = ser.readline()
#     # print('raw data:', data)
#     data = data.decode()
#     print('decoded data:', data)
#     # print('count', count)
#     count += 1
#     if data != 'helloa':
#         print('what\'s wrong with you?')

#     if count == 100000:
#         print('done!')
        