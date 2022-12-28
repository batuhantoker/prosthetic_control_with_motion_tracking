
from subprocess import *
import os, threading
import pandas as pd
import sklearn
import joblib
original_path=os.getcwd()
os.chdir(os.getcwd()+"/Dataset")
svm = joblib.load('svm.pkl')
logreg = joblib.load('logreg.pkl')
clf = joblib.load('clf.pkl')
lda = joblib.load('lda.pkl')
data_recorder_path = original_path + r"\stream_data.py"
command = r"py -2 " + data_recorder_path
os.chdir(original_path)
print(command)

def main():
    process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE,stdin=PIPE)
    while process.poll() is None:
        line = process.stdout.readline()
        #print len(line.split(","))
        data_to_predict=pd.DataFrame(line.split(","))
        data_clear= data_to_predict[:-1].transpose()
        #print (data_clear)
        if len(line.split(","))>400:
            pred1=lda.predict(data_clear)
            pred2 = svm.predict(data_clear)
            print "LDA prediction", pred1
            print "SVM prediction", pred2

main()
