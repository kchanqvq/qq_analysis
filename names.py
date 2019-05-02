import os
from datetime import datetime
result={}
with open("雾雨魔法店.txt","r") as infile:
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
            qqno=None
            if lastbra<=0:
                lastsharp = ln.rfind("<")
                if lastsharp <=0:
                    raise ValueError("l")
                else:
                    qqno=ln[lastsharp+1:-2]
                    result[qqno] = ln[secondspace+1:lastsharp]
                    #print(qqno)
            else:
                qqno = int(ln[lastbra+1:-2])
                result[qqno] = ln[secondspace+1:lastbra]
        except Exception as e:
            pass
        finally:
            ln = infile.readline()
with open("mk_names.txt","w") as outfile:
    outfile.write(str(result))
