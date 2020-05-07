import sys


with open(sys.argv[1]+"_TRUE.phy", "r") as file:
    num_data = 1
    skipped_header = False
    out = open("../A"+"{0:0=4d}".format(num_data)+".fa", "w")
    for line in file:
        splitln = line.split()
        if len(splitln)==0:
            if skipped_header:
                out.close()
                skipped_header = False
                num_data += 1
                out = open("../A"+"{0:0=4d}".format(num_data)+".fa", "w")
        else:
            if not skipped_header:
                skipped_header = True
            else:
                if len(splitln[1].replace("-","")) > 0:
                    out.write(">" + splitln[0] + "\n")
                    out.write(splitln[1] + "\n")
    out.close()
