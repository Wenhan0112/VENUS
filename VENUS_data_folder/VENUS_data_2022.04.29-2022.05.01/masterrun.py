import os
import time

counter=0
start_time=int(time.time())
for i in range(80):
    f=open('continuerunning','r')
    if int(f.readline())==1:
        print('starting #',counter+1,' at ',time.strftime("%d %b %Y %H:%M:%S",time.localtime()))
        os.system("python3 util_DataSheet.py")
        now_time=int(time.time())
        os.system("python3 tuning-wenhan_ms_dst.py > temp")
        os.rename('temp','dump_wenhan_'+str(now_time))
        print('Wenhan program time: %.1f min'%((time.time()-now_time)/60.0))
        f.seek(0)
    f.close()

    f=open('continuerunning','r')
    if int(f.readline())==1:
        os.system("python3 util_DataSheet.py")
        now_time=int(time.time())
        os.system("python3 tuning-harvey_dst_just3.py > temp")
        os.rename('temp','dump_harvey_'+str(now_time))
        print('Harvey program time: %.1f min'%((time.time()-now_time)/60.0))
        counter=counter+1
        f.seek(0)
    f.close()

    
