import pandas as pd
import sys
file_name = sys.argv[1]
df = pd.read_csv(file_name)
diff = df['to'] - df['from']
print("Increases persent {:.1%}".format(((diff > 0).mean()),))
print("Mean difference %f" % diff.mean())
print("Var difference %f" % diff.var())
