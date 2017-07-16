#!/usr/bin/env python
###########################################################
'''
    Copyright 2017 

    This file is part of Algo LSTM.
    
'''
###########################################################
# Written by Tim Ioannidis & Arman Rezaee  
###########################################################

##########################################################
############  Main script that coordinates  ##############
#############  all parts of the program   ################
##########################################################

import sys, argparse, os, platform, subprocess

def mybash(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = []
    while True:
        line = p.stdout.readline()
        sys.stdout.write(line)
        stdout.append(line)
        if line == '' and p.poll() != None:
            break
    return ''.join(stdout)

if __name__ == '__main__':
    with open('industries.txt','r') as f:
        d = f.read()
    folders = d.splitlines()
    with open('rmse.txt','r') as f:
        dat = f.read()
    for ii,folder in enumerate(folders):
        if (folder not in dat):
            cmd = "./main.py -d "+folder+" -t"
            print("Running folder "+folder+" ("+str(ii+1)+"/"+str(len(folders))+")")
            s = mybash(cmd)
        else:
            print("Folder "+folder+ " already trained. Skipping.")
