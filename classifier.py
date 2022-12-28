import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
path=os.chdir(os.getcwd()+"/Dataset")

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
print(all_filenames)
variable_names = ['left_palm_position_x', ' left_palm_position_y', ' left_palm_position_z', ' left_palm_normal_x',
                  '                                  left_palm_normal_y', ' left_palm_normal_z',
                  ' left_hand_direction_x', ' left_hand_direction_y',
                  '                                  left_hand_direction_z', ' left_palm_velocity_x',
                  ' left_palm_velocity_y', '                                  left_palm_velocity_z', ' left_hand_pitch',
                  ' left_hand_roll', ' left_hand_yaw', '                                  left_arm_direction_x',
                  ' left_arm_direction_y', ' left_arm_direction_z',
                  '                                  left_wrist_position_x', ' left_wrist_position_y',
                  ' left_wrist_position_z', '                                  left_elbow_position_x',
                  ' left_elbow_position_y', ' left_elbow_position_z',
                  '                                  left_thumb_length', ' left_thumb_width',
                  ' left_thumb_metacarpal_start_x', '                                  left_thumb_metacarpal_start_y',
                  ' left_thumb_metacarpal_start_z', '                                  left_thumb_metacarpal_end_x',
                  ' left_thumb_metacarpal_end_y', ' left_thumb_metacarpal_end_z',
                  '                                  left_thumb_metacarpal_direction_x',
                  ' left_thumb_metacarpal_direction_y',
                  '                                  left_thumb_metacarpal_direction_z', ' left_thumb_proximal_start_x',
                  '                                  left_thumb_proximal_start_y', ' left_thumb_proximal_start_z',
                  ' left_thumb_proximal_end_x', '                                  left_thumb_proximal_end_y',
                  ' left_thumb_proximal_end_z', ' left_thumb_proximal_direction_x',
                  '                                  left_thumb_proximal_direction_y',
                  ' left_thumb_proximal_direction_z',
                  '                                  left_thumb_intermediate_start_x',
                  ' left_thumb_intermediate_start_y',
                  '                                  left_thumb_intermediate_start_z', ' left_thumb_intermediate_end_x',
                  '                                  left_thumb_intermediate_end_y', ' left_thumb_intermediate_end_z',
                  '                                  left_thumb_intermediate_direction_x',
                  ' left_thumb_intermediate_direction_y',
                  '                                  left_thumb_intermediate_direction_z', ' left_thumb_distal_start_x',
                  '                                  left_thumb_distal_start_y', ' left_thumb_distal_start_z',
                  ' left_thumb_distal_end_x', '                                  left_thumb_distal_end_y',
                  ' left_thumb_distal_end_z', ' left_thumb_distal_direction_x',
                  '                                  left_thumb_distal_direction_y', ' left_thumb_distal_direction_z',
                  ' left_index_length', '                                  left_index_width',
                  ' left_index_metacarpal_start_x', ' left_index_metacarpal_start_y',
                  '                                  left_index_metacarpal_start_z', ' left_index_metacarpal_end_x',
                  '                                  left_index_metacarpal_end_y', ' left_index_metacarpal_end_z',
                  '                                  left_index_metacarpal_direction_x',
                  ' left_index_metacarpal_direction_y',
                  '                                  left_index_metacarpal_direction_z', ' left_index_proximal_start_x',
                  '                                  left_index_proximal_start_y', ' left_index_proximal_start_z',
                  ' left_index_proximal_end_x', '                                  left_index_proximal_end_y',
                  ' left_index_proximal_end_z', ' left_index_proximal_direction_x',
                  '                                  left_index_proximal_direction_y',
                  ' left_index_proximal_direction_z',
                  '                                  left_index_intermediate_start_x',
                  ' left_index_intermediate_start_y',
                  '                                  left_index_intermediate_start_z', ' left_index_intermediate_end_x',
                  '                                  left_index_intermediate_end_y', ' left_index_intermediate_end_z',
                  '                                  left_index_intermediate_direction_x',
                  ' left_index_intermediate_direction_y',
                  '                                  left_index_intermediate_direction_z', ' left_index_distal_start_x',
                  '                                  left_index_distal_start_y', ' left_index_distal_start_z',
                  ' left_index_distal_end_x', '                                  left_index_distal_end_y',
                  ' left_index_distal_end_z', ' left_index_distal_direction_x',
                  '                                  left_index_distal_direction_y', ' left_index_distal_direction_z',
                  ' left_middle_length', '                                  left_middle_width',
                  ' left_middle_metacarpal_start_x', ' left_middle_metacarpal_start_y',
                  '                                  left_middle_metacarpal_start_z', ' left_middle_metacarpal_end_x',
                  '                                  left_middle_metacarpal_end_y', ' left_middle_metacarpal_end_z',
                  '                                  left_middle_metacarpal_direction_x',
                  ' left_middle_metacarpal_direction_y',
                  '                                  left_middle_metacarpal_direction_z',
                  ' left_middle_proximal_start_x', '                                  left_middle_proximal_start_y',
                  ' left_middle_proximal_start_z', '                                  left_middle_proximal_end_x',
                  ' left_middle_proximal_end_y', ' left_middle_proximal_end_z',
                  '                                  left_middle_proximal_direction_x',
                  ' left_middle_proximal_direction_y',
                  '                                  left_middle_proximal_direction_z',
                  ' left_middle_intermediate_start_x',
                  '                                  left_middle_intermediate_start_y',
                  ' left_middle_intermediate_start_z',
                  '                                  left_middle_intermediate_end_x', ' left_middle_intermediate_end_y',
                  '                                  left_middle_intermediate_end_z',
                  ' left_middle_intermediate_direction_x',
                  '                                  left_middle_intermediate_direction_y',
                  ' left_middle_intermediate_direction_z',
                  '                                  left_middle_distal_start_x', ' left_middle_distal_start_y',
                  ' left_middle_distal_start_z', '                                  left_middle_distal_end_x',
                  ' left_middle_distal_end_y', ' left_middle_distal_end_z',
                  '                                  left_middle_distal_direction_x', ' left_middle_distal_direction_y',
                  '                                  left_middle_distal_direction_z', ' left_ring_length',
                  ' left_ring_width', '                                  left_ring_metacarpal_start_x',
                  ' left_ring_metacarpal_start_y', '                                  left_ring_metacarpal_start_z',
                  ' left_ring_metacarpal_end_x', ' left_ring_metacarpal_end_y',
                  '                                  left_ring_metacarpal_end_z', ' left_ring_metacarpal_direction_x',
                  '                                  left_ring_metacarpal_direction_y',
                  ' left_ring_metacarpal_direction_z', '                                  left_ring_proximal_start_x',
                  ' left_ring_proximal_start_y', ' left_ring_proximal_start_z',
                  '                                  left_ring_proximal_end_x', ' left_ring_proximal_end_y',
                  ' left_ring_proximal_end_z', '                                  left_ring_proximal_direction_x',
                  ' left_ring_proximal_direction_y', '                                  left_ring_proximal_direction_z',
                  ' left_ring_intermediate_start_x', '                                  left_ring_intermediate_start_y',
                  ' left_ring_intermediate_start_z', '                                  left_ring_intermediate_end_x',
                  ' left_ring_intermediate_end_y', '                                  left_ring_intermediate_end_z',
                  ' left_ring_intermediate_direction_x',
                  '                                  left_ring_intermediate_direction_y',
                  ' left_ring_intermediate_direction_z', '                                  left_ring_distal_start_x',
                  ' left_ring_distal_start_y', ' left_ring_distal_start_z',
                  '                                  left_ring_distal_end_x', ' left_ring_distal_end_y',
                  ' left_ring_distal_end_z', '                                  left_ring_distal_direction_x',
                  ' left_ring_distal_direction_y', '                                  left_ring_distal_direction_z',
                  ' left_pinky_length', ' left_pinky_width',
                  '                                  left_pinky_metacarpal_start_x', ' left_pinky_metacarpal_start_y',
                  '                                  left_pinky_metacarpal_start_z', ' left_pinky_metacarpal_end_x',
                  '                                  left_pinky_metacarpal_end_y', ' left_pinky_metacarpal_end_z',
                  '                                  left_pinky_metacarpal_direction_x',
                  ' left_pinky_metacarpal_direction_y',
                  '                                  left_pinky_metacarpal_direction_z', ' left_pinky_proximal_start_x',
                  '                                  left_pinky_proximal_start_y', ' left_pinky_proximal_start_z',
                  ' left_pinky_proximal_end_x', '                                  left_pinky_proximal_end_y',
                  ' left_pinky_proximal_end_z', ' left_pinky_proximal_direction_x',
                  '                                  left_pinky_proximal_direction_y',
                  ' left_pinky_proximal_direction_z',
                  '                                  left_pinky_intermediate_start_x',
                  ' left_pinky_intermediate_start_y',
                  '                                  left_pinky_intermediate_start_z', ' left_pinky_intermediate_end_x',
                  '                                  left_pinky_intermediate_end_y', ' left_pinky_intermediate_end_z',
                  '                                  left_pinky_intermediate_direction_x',
                  ' left_pinky_intermediate_direction_y',
                  '                                  left_pinky_intermediate_direction_z', ' left_pinky_distal_start_x',
                  '                                  left_pinky_distal_start_y', ' left_pinky_distal_start_z',
                  ' left_pinky_distal_end_x', '                                  left_pinky_distal_end_y',
                  ' left_pinky_distal_end_z', ' left_pinky_distal_direction_x',
                  '                                  left_pinky_distal_direction_y', ' left_pinky_distal_direction_z',
                  ' right_palm_position_x', '                                  right_palm_position_y',
                  ' right_palm_position_z', ' right_palm_normal_x',
                  '                                  right_palm_normal_y', ' right_palm_normal_z',
                  ' right_hand_direction_x', '                                  right_hand_direction_y',
                  ' right_hand_direction_z', ' right_palm_velocity_x',
                  '                                  right_palm_velocity_y', ' right_palm_velocity_z',
                  ' right_hand_pitch', ' right_hand_roll', '                                  right_hand_yaw',
                  ' right_arm_direction_x', ' right_arm_direction_y', ' right_arm_direction_z',
                  '                                  right_wrist_position_x', ' right_wrist_position_y',
                  ' right_wrist_position_z', '                                  right_elbow_position_x',
                  ' right_elbow_position_y', ' right_elbow_position_z',
                  '                                  right_thumb_length', ' right_thumb_width',
                  ' right_thumb_metacarpal_start_x', '                                  right_thumb_metacarpal_start_y',
                  ' right_thumb_metacarpal_start_z', '                                  right_thumb_metacarpal_end_x',
                  ' right_thumb_metacarpal_end_y', '                                  right_thumb_metacarpal_end_z',
                  ' right_thumb_metacarpal_direction_x',
                  '                                  right_thumb_metacarpal_direction_y',
                  ' right_thumb_metacarpal_direction_z',
                  '                                  right_thumb_proximal_start_x', ' right_thumb_proximal_start_y',
                  '                                  right_thumb_proximal_start_z', ' right_thumb_proximal_end_x',
                  ' right_thumb_proximal_end_y', '                                  right_thumb_proximal_end_z',
                  ' right_thumb_proximal_direction_x',
                  '                                  right_thumb_proximal_direction_y',
                  ' right_thumb_proximal_direction_z',
                  '                                  right_thumb_intermediate_start_x',
                  ' right_thumb_intermediate_start_y',
                  '                                  right_thumb_intermediate_start_z',
                  ' right_thumb_intermediate_end_x', '                                  right_thumb_intermediate_end_y',
                  ' right_thumb_intermediate_end_z',
                  '                                  right_thumb_intermediate_direction_x',
                  ' right_thumb_intermediate_direction_y',
                  '                                  right_thumb_intermediate_direction_z',
                  ' right_thumb_distal_start_x', '                                  right_thumb_distal_start_y',
                  ' right_thumb_distal_start_z', ' right_thumb_distal_end_x',
                  '                                  right_thumb_distal_end_y', ' right_thumb_distal_end_z',
                  ' right_thumb_distal_direction_x', '                                  right_thumb_distal_direction_y',
                  ' right_thumb_distal_direction_z', ' right_index_length',
                  '                                  right_index_width', ' right_index_metacarpal_start_x',
                  ' right_index_metacarpal_start_y', '                                  right_index_metacarpal_start_z',
                  ' right_index_metacarpal_end_x', '                                  right_index_metacarpal_end_y',
                  ' right_index_metacarpal_end_z',
                  '                                  right_index_metacarpal_direction_x',
                  ' right_index_metacarpal_direction_y',
                  '                                  right_index_metacarpal_direction_z',
                  ' right_index_proximal_start_x', '                                  right_index_proximal_start_y',
                  ' right_index_proximal_start_z', '                                  right_index_proximal_end_x',
                  ' right_index_proximal_end_y', ' right_index_proximal_end_z',
                  '                                  right_index_proximal_direction_x',
                  ' right_index_proximal_direction_y',
                  '                                  right_index_proximal_direction_z',
                  ' right_index_intermediate_start_x',
                  '                                  right_index_intermediate_start_y',
                  ' right_index_intermediate_start_z',
                  '                                  right_index_intermediate_end_x', ' right_index_intermediate_end_y',
                  '                                  right_index_intermediate_end_z',
                  ' right_index_intermediate_direction_x',
                  '                                  right_index_intermediate_direction_y',
                  ' right_index_intermediate_direction_z',
                  '                                  right_index_distal_start_x', ' right_index_distal_start_y',
                  ' right_index_distal_start_z', '                                  right_index_distal_end_x',
                  ' right_index_distal_end_y', ' right_index_distal_end_z',
                  '                                  right_index_distal_direction_x', ' right_index_distal_direction_y',
                  '                                  right_index_distal_direction_z', ' right_middle_length',
                  ' right_middle_width', '                                  right_middle_metacarpal_start_x',
                  ' right_middle_metacarpal_start_y',
                  '                                  right_middle_metacarpal_start_z', ' right_middle_metacarpal_end_x',
                  '                                  right_middle_metacarpal_end_y', ' right_middle_metacarpal_end_z',
                  '                                  right_middle_metacarpal_direction_x',
                  ' right_middle_metacarpal_direction_y',
                  '                                  right_middle_metacarpal_direction_z',
                  ' right_middle_proximal_start_x', '                                  right_middle_proximal_start_y',
                  ' right_middle_proximal_start_z', '                                  right_middle_proximal_end_x',
                  ' right_middle_proximal_end_y', ' right_middle_proximal_end_z',
                  '                                  right_middle_proximal_direction_x',
                  ' right_middle_proximal_direction_y',
                  '                                  right_middle_proximal_direction_z',
                  ' right_middle_intermediate_start_x',
                  '                                  right_middle_intermediate_start_y',
                  ' right_middle_intermediate_start_z',
                  '                                  right_middle_intermediate_end_x',
                  ' right_middle_intermediate_end_y',
                  '                                  right_middle_intermediate_end_z',
                  ' right_middle_intermediate_direction_x',
                  '                                  right_middle_intermediate_direction_y',
                  ' right_middle_intermediate_direction_z',
                  '                                  right_middle_distal_start_x', ' right_middle_distal_start_y',
                  ' right_middle_distal_start_z', '                                  right_middle_distal_end_x',
                  ' right_middle_distal_end_y', ' right_middle_distal_end_z',
                  '                                  right_middle_distal_direction_x',
                  ' right_middle_distal_direction_y',
                  '                                  right_middle_distal_direction_z', ' right_ring_length',
                  ' right_ring_width', '                                  right_ring_metacarpal_start_x',
                  ' right_ring_metacarpal_start_y', '                                  right_ring_metacarpal_start_z',
                  ' right_ring_metacarpal_end_x', '                                  right_ring_metacarpal_end_y',
                  ' right_ring_metacarpal_end_z', '                                  right_ring_metacarpal_direction_x',
                  ' right_ring_metacarpal_direction_y',
                  '                                  right_ring_metacarpal_direction_z', ' right_ring_proximal_start_x',
                  '                                  right_ring_proximal_start_y', ' right_ring_proximal_start_z',
                  ' right_ring_proximal_end_x', '                                  right_ring_proximal_end_y',
                  ' right_ring_proximal_end_z', ' right_ring_proximal_direction_x',
                  '                                  right_ring_proximal_direction_y',
                  ' right_ring_proximal_direction_z',
                  '                                  right_ring_intermediate_start_x',
                  ' right_ring_intermediate_start_y',
                  '                                  right_ring_intermediate_start_z', ' right_ring_intermediate_end_x',
                  '                                  right_ring_intermediate_end_y', ' right_ring_intermediate_end_z',
                  '                                  right_ring_intermediate_direction_x',
                  ' right_ring_intermediate_direction_y',
                  '                                  right_ring_intermediate_direction_z', ' right_ring_distal_start_x',
                  '                                  right_ring_distal_start_y', ' right_ring_distal_start_z',
                  ' right_ring_distal_end_x', '                                  right_ring_distal_end_y',
                  ' right_ring_distal_end_z', ' right_ring_distal_direction_x',
                  '                                  right_ring_distal_direction_y', ' right_ring_distal_direction_z',
                  ' right_pinky_length', '                                  right_pinky_width',
                  ' right_pinky_metacarpal_start_x', ' right_pinky_metacarpal_start_y',
                  '                                  right_pinky_metacarpal_start_z', ' right_pinky_metacarpal_end_x',
                  '                                  right_pinky_metacarpal_end_y', ' right_pinky_metacarpal_end_z',
                  '                                  right_pinky_metacarpal_direction_x',
                  ' right_pinky_metacarpal_direction_y',
                  '                                  right_pinky_metacarpal_direction_z',
                  ' right_pinky_proximal_start_x', '                                  right_pinky_proximal_start_y',
                  ' right_pinky_proximal_start_z', '                                  right_pinky_proximal_end_x',
                  ' right_pinky_proximal_end_y', ' right_pinky_proximal_end_z',
                  '                                  right_pinky_proximal_direction_x',
                  ' right_pinky_proximal_direction_y',
                  '                                  right_pinky_proximal_direction_z',
                  ' right_pinky_intermediate_start_x',
                  '                                  right_pinky_intermediate_start_y',
                  ' right_pinky_intermediate_start_z',
                  '                                  right_pinky_intermediate_end_x', ' right_pinky_intermediate_end_y',
                  '                                  right_pinky_intermediate_end_z',
                  ' right_pinky_intermediate_direction_x',
                  '                                  right_pinky_intermediate_direction_y',
                  ' right_pinky_intermediate_direction_z',
                  '                                  right_pinky_distal_start_x', ' right_pinky_distal_start_y',
                  ' right_pinky_distal_start_z', '                                  right_pinky_distal_end_x',
                  ' right_pinky_distal_end_y', ' right_pinky_distal_end_z',
                  '                                  right_pinky_distal_direction_x', ' right_pinky_distal_direction_y',
                  ' right_pinky_distal_direction_z', 'move']

print(len(variable_names))
df_append = pd.DataFrame()
#append all files together
for file in all_filenames:
    df_temp = pd.read_csv(file)
    df_temp.to_csv(file, header=variable_names, index=False)
for file in all_filenames:
    df_temp = pd.read_csv(file)
    df_append = df_append.append(df_temp)
df_append

# 6. Single DF is saved to the path in CSV format, without index column
df_append.to_csv( 'all_combined.csv', index=False)
dataset=pd.read_csv('all_combined.csv')
dataset['move'] = pd.factorize(dataset['move'])[0] + 1
y=dataset.iloc[:,-1:]
X=dataset.iloc[:,:-1]
print(X.shape,y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
     .format(lda.score(X_test, y_test)))


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))

import pickle
with open('svm.pkl', 'wb') as f:
    pickle.dump(svm, f)
with open('clf.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('lda.pkl', 'wb') as f:
    pickle.dump(lda, f)
with open('logreg.pkl', 'wb') as f:
    pickle.dump(logreg, f)