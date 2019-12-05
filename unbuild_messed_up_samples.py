import os
from os import walk
import random 
import numpy as np
#the following code will only work on macs because I got very lazy with path conventions.
#todo: fix by always using os.join() instead of simple string concatenation

def ReadList(list_file):
 f=open(list_file,"r")
 lines=f.readlines()
 list_sig=[]
 for x in lines:
    list_sig.append(x.rstrip())
 f.close()
 return list_sig

filenames = ReadList("CREMAD_dev.scp")

projdir = os.getcwd() #start out in Samples folder
wrongpath = projdir + "/Train/" 
goalpath = projdir + "/Dev/"

emolist = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
for emo in emolist:
    os.chdir(wrongpath+emo+"/")
    emofilelist = os.listdir() 
    for f in emofilelist:
        if f in filenames:
            os.rename(wrongpath +emo+"/"+f,goalpath +emo+"/"+f)

    


