#!/bin/env python3
""" Do a CSD after asking user for min&max currents, number of steps, and filename"""


import inspect
import itertools
import time
import types
import signal
import numpy as np
import datetime

import venus_utils.venusplc as venusplc


venus = venusplc.VENUSController('/home/damon/venus_data_project/config/config.ini')

def get_csd(imin,imax,nsteps,writefile):
  starttime = time.time()
  ibatman = venus.read(['batman_i'])
  ipts=np.sqrt(np.linspace(imin*imin,imax*imax,num=nsteps))
  writefile.write("%i\n"%(nsteps))
  venus.setpoint({'batman_i':ipts[0]})
  time.sleep(2)
  for i in range(nsteps):
          istart = venus.read(['batman_i'])
          trynum = 0
          inow = 0.0
          while(inow!=istart and trynum<10):
             venus.write({'batman_i':ipts[i]})
             inow = venus.read(['batman_i'])
             trynum=trynum+1
          print(i,trynum)
          writefile.write("%7.3f %7.3f %7.3f %12.3f\n"%(venus.read(['m_over_q']),ipts[i],venus.read(['batman_i']),venus.read(['fcv1_i'])*1e6))
          if i==19: 
              print("20 done. time remaining in seconds: ",(time.time()-starttime)*(nsteps-20)/20.)
  venus.setpoint({'batman_i':ibatman})


#filename = str(input('enter filename to write CSD or press enter for "tempCSD": '))
#if len(filename==0): filename='tempCSD'
#writefile=open(filename,'w')
starttimesec = str(int(time.time()))
writefile=open('csd_'+starttimesec,'w')
#imin = float(input('minimum batman current for CSD in amps: '))
#imax = float(input('maximum batman current for CSD in amps: '))
#numsteps = int(input('number of steps in current: '))
imin=59.0
imax=97.0
numsteps=500

get_csd(imin,imax,numsteps,writefile)
writefile.close()

