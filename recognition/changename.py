import os
Root_dir = r"G:/TestData"
files = os.listdir(Root_dir)
for file in files:
    newname = '{:04d}.mat'.format(int(file[:-10]))
    newname = os.path.join(Root_dir, newname)
    oldname = os.path.join(Root_dir, file)
    os.rename(oldname, newname)