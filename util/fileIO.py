#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/6/21

import string,re,os


def write_string(s, output_path):
    with open(output_path, 'w') as output_file:
        output_file.write(s)


def read_file(input_path):
    with open(input_path, 'r', errors='replace_with_space') as input_file:
        return input_file.read()



# if __name__ == '__main__':
#

#     print("OK")



