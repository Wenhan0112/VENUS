import os
import time

counter=0
start_time=int(time.time())
for i in range(80):
    f=open('continuerunning','r')
    if int(f.readline())==1:
        os.system("python3 util_DataSheet.py")
        now_time=int(time.time())
        os.system("python3 tuning_harvey_04.py > temp")
        os.rename('temp','dump_harvey_'+str(now_time))
        print('Harvey program time: %.1f min'%((time.time()-now_time)/60.0))
        counter=counter+1
        f.seek(0)
    f.close()

    
