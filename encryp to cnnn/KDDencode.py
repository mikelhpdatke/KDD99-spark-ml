import sys
import  json
if __name__=="__main__":
    print(sys.argv)
    fi = sys.argv[1]
    with open(fi) as f:
        content = f.readlines()
    content =[x.strip() for x in content]

    dict = {}

    count = 1
    for x in content:
        list = str(x).split(',')

        dict[list[1]] = count

        dict[list[2]] = count

        dict[list[3]] = count

        dict[list[len(list) - 1]] = count
    ans = {}
    count = 100
    for x, y in dict.items():
        print(x, y)
        ans[x] = str(count)
        count = count + 1
    with open('table.code', "w+") as f:
        json.dump(ans, f)
    print("Save table successfully to table.code")
    with open(sys.argv[2], "w+") as f:
        for x in content:
            list = str(x).split(',')
            list[1] = ans[list[1]]
            list[2] = ans[list[2]]
            list[3] = ans[list[3]]
            list[len(list) - 1] = ans[list[len(list) - 1]]
            s = ""
            for i in list:
                s = s + str(i) + ','
            f.write(s[:len(s) - 1] + "\n")