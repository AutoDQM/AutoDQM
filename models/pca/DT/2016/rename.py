import os 

for filename in os.listdir('.'):
    if 'rename' in filename:
        continue
    dst = f'Segment_{filename}'
    os.rename(filename, dst)
