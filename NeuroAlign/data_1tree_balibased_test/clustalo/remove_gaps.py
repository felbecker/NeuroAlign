import sys

for i in range(int(sys.argv[1])):
    with open("../A"+"{0:0=4d}".format(i+1)+".fa", "r") as file:
        content = file.read().replace('-', '')
        with open("A"+"{0:0=4d}".format(i+1)+".fa", "w") as out_file:
            out_file.write(content)
