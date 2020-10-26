import sys
import json
if __name__=="__main__":
    ans = json.load(open("table.code"))
    print(type(ans))
    rev = {}
    for x, y in ans.items():
        rev[y] = x
    with open(sys.argv[1], "r+") as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        with open(sys.argv[2], "w+") as of:
            for x in content:
                list = str(x).split(',')
                list[1] = rev[list[1]]
                list[2] = rev[list[2]]
                list[3] = rev[list[3]]
                list[len(list) - 1] = rev[list[len(list) - 1]]
                s = ""
                for i in list:
                    s = s + str(i) + ','
                of.write(s[:len(s) - 1] + "\n")