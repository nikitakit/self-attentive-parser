def proc_line(line):
    line = line.strip()
    line = "".join(line.split("##")[::2])
    return line

if __name__ == "__main__":
    import fileinput
    for line in fileinput.input():
        print(proc_line(line))
