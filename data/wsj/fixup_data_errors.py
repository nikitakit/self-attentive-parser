"""
Fixes three errors in the training data:
- One tree in LDC2015T13 where the sentence-final period is attached to the
  dummy root node (sometimes called TOP) instead of the top-level S node. This
  error does not occur in the original treebank.
- One tree tokenizes underlying 'Tis as "'T - is" (extra -)
- One tree tokenizes underlying I'm-coming as "I - 'm coming" (transposition)

The first error leads to scripts crashing because it violates the assumption of
always having a single root node below TOP.

The others lead to errors when attempting to match the parsed trees to raw
text, which we do in order to recover the original whitespace in the data.
"""

A_ORIG = "(PRP 'T)) (HYPH -) (VP (VBZ is)"
A_REPLACEMENT = "(PRP 'T)) (VP (VBZ is)"
B_ORIG = "(PRP I)) (HYPH -) (VP (VBP 'm)"
B_REPLACEMENT = "(PRP I)) (VP (VBP 'm) (HYPH -)"

def proc_line(line):
    stripped = line.strip()
    
    if stripped.startswith("( (") and stripped.endswith(") (. .))"):
        return stripped[2:-len(") (. .))")] + " (. .))"
    elif A_ORIG in stripped:
        return stripped.replace(A_ORIG, A_REPLACEMENT)
    elif B_ORIG in stripped:
        return stripped.replace(B_ORIG, B_REPLACEMENT)
    else:
        return stripped


if __name__ == "__main__":
    import fileinput
    for line in fileinput.input():
        print(proc_line(line))
