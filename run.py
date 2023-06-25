import os


n_estimators=[100,120,150,180,200,250]

max_depth=[5,10,15,20,25]


for n in n_estimators:
    for m in max_depth:
        os.system(f"python app3.py -n{n} -m{m}")