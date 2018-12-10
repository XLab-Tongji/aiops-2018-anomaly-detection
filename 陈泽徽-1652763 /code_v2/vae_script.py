import os

for i in range(26):
    os.system('python vae_generate.py '+str(i+1))