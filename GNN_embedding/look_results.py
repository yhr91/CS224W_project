import ast
import matplotlib.pyplot as plt
import os

for fname in os.listdir('.'):
    if fname[-4:] != '.txt': continue
    with open(fname, 'r') as f:
        content = f.read()
        losses, vals = ast.literal_eval(content)
    plt.plot(losses, label='losses ' + fname)
    plt.plot(vals, label='vals ' + fname)
plt.legend()
plt.show()