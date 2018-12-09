import os

with open('result.txt','w') as f:
    pass

with open('output.csv','w') as f:
    pass

for i in range(26):
    os.system('python dnn.py '+str(i+1))