import os
from datetime import datetime
result={}
with open("复活的百度物理吧唯一.txt","r") as infile:
    ln = infile.readline()
    count = 0
    while ln:
        count += 1
        if count%10000 == 0:
            print("Processed ",count," lines")
        try:
            firstspace = ln.find(" ")
            if firstspace<=0:
                raise ValueError("f")
            secondspace = ln.find(" ",firstspace+1)
            if secondspace<=0:
                raise ValueError("s")
            date = datetime.strptime(ln[:secondspace], '%Y-%m-%d %H:%M:%S').timestamp()
            lastbra = ln.rfind("(")
            if lastbra<=0:
                raise ValueError("l")
            qqno = int(ln[lastbra+1:-2])
            result[qqno] = ln[secondspace+1:lastbra]
        except Exception as e:
            pass
        finally:
            ln = infile.readline()
with open("names.txt","w") as outfile:
    outfile.write(str(result))
