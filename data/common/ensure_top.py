def proc_line(line):
    stripped = line.strip()
    
    if not stripped.startswith("(TOP"):
        return "(TOP {})".format(stripped)
    else:
        return stripped

if __name__ == "__main__":
    import fileinput
    for line in fileinput.input():
        print(proc_line(line))
