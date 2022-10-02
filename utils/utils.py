
"""
The MIT License

Copyright (c) 2022 SGBS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import time
import sys
import os
from datetime import datetime
import logging
import logging.config
import numpy as np


process_start_time = datetime.now()
result_folder = './result/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'


def get_result_folder():
    return result_folder


def set_result_folder(folder):
    global result_folder
    result_folder = folder


def create_logger(log_file=None):
    if 'filepath' not in log_file:
        log_file['filepath'] = get_result_folder()

    if 'desc' in log_file:
        log_file['filepath'] = log_file['filepath'].format(desc='_' + log_file['desc'])
    else:
        log_file['filepath'] = log_file['filepath'].format(desc='')

    set_result_folder(log_file['filepath'])

    if 'filename' in log_file:
        filename = log_file['filepath'] + '/' + log_file['filename']
    else:
        filename = log_file['filepath'] + '/' + 'log.txt'

    if not os.path.exists(log_file['filepath']):
        os.makedirs(log_file['filepath'])

    file_mode = 'a' if os.path.isfile(filename)  else 'w'

    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # write to file
    fileout = logging.FileHandler(filename, mode=file_mode)
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0


class LogData:
    def __init__(self):
        self.keys = set()
        self.data = {}

    def get_raw_data(self):
        return self.keys, self.data

    def set_raw_data(self, r_data):
        self.keys, self.data = r_data

    def append_all(self, key, *args):
        if len(args) == 1:
            value = [list(range(len(args[0]))), args[0]]
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            self.data[key].extend(value)
        else:
            self.data[key] = np.stack(value, axis=1).tolist()
            self.keys.add(key)

    def append(self, key, *args):
        if len(args) == 1:
            args = args[0]

            if isinstance(args, int) or isinstance(args, float):
                if self.has_key(key):
                    value = [len(self.data[key]), args]
                else:
                    value = [0, args]
            elif type(args) == tuple:
                value = list(args)
            elif type(args) == list:
                value = args
            else:
                raise ValueError('Unsupported value type')
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            self.data[key].append(value)
        else:
            self.data[key] = [value]
            self.keys.add(key)

    def get_last(self, key):
        if not self.has_key(key):
            return None
        return self.data[key][-1]

    def has_key(self, key):
        return key in self.keys

    def get(self, key):
        split = np.hsplit(np.array(self.data[key]), 2)

        return split[1].squeeze().tolist()

    def getXY(self, key, start_idx=0):
        split = np.hsplit(np.array(self.data[key]), 2)

        xs = split[0].squeeze().tolist()
        ys = split[1].squeeze().tolist()

        if type(xs) is not list:
            return xs, ys

        if start_idx == 0:
            return xs, ys
        elif start_idx in xs:
            idx = xs.index(start_idx)
            return xs[idx:], ys[idx:]
        else:
            raise KeyError('no start_idx value in X axis data.')

    def get_keys(self):
        return self.keys


class TimeEstimator:
    def __init__(self):
        self.logger = logging.getLogger('TimeEstimator')
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count-1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total-count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time*60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time*60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            count, total, elapsed_time_str, remain_time_str))


def util_print_log_array(logger, result_log: LogData):
    assert type(result_log) == LogData, 'use LogData Class for result_log.'

    for key in result_log.get_keys():
        logger.info('{} = {}'.format(key+'_list', result_log.get(key)))




