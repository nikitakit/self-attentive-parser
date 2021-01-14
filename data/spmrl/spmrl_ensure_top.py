def proc_line(line):
    stripped = line.strip()

    if stripped.startswith("(ROOT "):
        return "(TOP " + stripped[len("(ROOT "):]
    elif stripped.startswith("( ("):
        return "(TOP " + stripped[len("( "):]
    elif stripped.startswith("(VROOT "):
        return "(TOP " + stripped[len("(VROOT "):]
    elif not stripped.startswith("(TOP"):
        return "(TOP {})".format(stripped)
    else:
        return stripped

if __name__ == "__main__":
    import fileinput
    for line in fileinput.input():
        print(proc_line(line))
