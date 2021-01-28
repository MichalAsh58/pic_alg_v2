import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc, classification_report, mean_squared_error,mean_absolute_error,mean_squared_error,f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from xgboost import XGBClassifier
import keras.backend as K
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import math
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import os
import json
import joblib
import tensorflow_addons as tfa
import gym
from collections import Counter

from Custum_metric import ConfusionMatrixMetric
from Custom_loss import WeightedBinaryCrossEntropy
from tables_creation import EAT_Table,LEAP_db,Katz_db,CARE_data

def Type(type,df):
    col_names=["FA_Egg","FA_Milk","FA_Peanut","FA_general","SCORAD"]
    y=df[type]
    X=df.drop(columns=col_names)
    fullname = type.replace("FA", "Food Allergy")
    fullname = fullname.replace("_", " ")
    fullname = fullname.replace("SCORAD", "Atopic Dermatitis")

def Savemodel_DNN(model,name):
    path=f'/home/michal/MYOR Dropbox/R&D/Allergies Product Development/Prediction/Algorithm_Beta/24_01_2021_models/{datetime.datetime.now()}-DNN-{name}'
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    model.save(f'{path}/model')
    return path

def Savemodel_RF(model,name):
    path=f'/home/michal/MYOR Dropbox/R&D/Allergies Product Development/Prediction/Algorithm_Beta/24_01_2021_models/{datetime.datetime.now()}-RF-{name}'
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    joblib.dump(model, f'{path}/model')
    return path

def merge_tables(path1,path2,path3):
    df1=pd.read_excel(path1)
    df2=pd.read_excel(path2)
    df3=pd.read_excel(path3)
    # test
    DF=pd.concat([df1,df2,df3])
    print("merged table",DF.shape)
    # DF.to_excel("/home/michal/MYOR Dropbox/R&D/Partnerships/Tables/MERGED.xlsx",index=False)
    return DF

def create_tables(run_tables_creation,FA):
    if run_tables_creation:
        path_EAT = "/home/michal/MYOR Dropbox/R&D/Partnerships/EAT/EAT_Risk_Score.xlsx"
        path_LEAP = "/home/michal/MYOR Dropbox/R&D/Partnerships/LEAP/LEAP_Data.xlsx"
        path_KATZ = "/home/michal/MYOR Dropbox/R&D/Partnerships/Katz_Study/Output dat_myor_milk_Oct    4_2020.xlsx"
        EAT_Table(path_EAT,FA)
        # EAT_Table(path_EAT,merged=False)
        LEAP_db(path_LEAP,FA)
        Katz_db(path_KATZ,FA)

    if FA:
        merged_df=merge_tables("/home/michal/MYOR Dropbox/R&D/Partnerships/Tables/EAT_FA.xlsx","/home/michal/MYOR Dropbox/R&D/Partnerships/Tables/LEAP_FA.xlsx","/home/michal/MYOR Dropbox/R&D/Partnerships/Tables/KATZ_FA.xlsx")
    else:
        merged_df=merge_tables("/home/michal/MYOR Dropbox/R&D/Partnerships/Tables/EAT_AD.xlsx","/home/michal/MYOR Dropbox/R&D/Partnerships/Tables/LEAP_AD.xlsx","/home/michal/MYOR Dropbox/R&D/Partnerships/Tables/KATZ_AD.xlsx")
    return merged_df

def SVM_logisticRedression_model(X_train, X_test, y_train, y_test):
    Accuracy_LR=[]
    false_LR=[]
    AUC_LR=[]
    Accuracy_SVM=[]
    false_SVM=[]
    AUC_SVM=[]
    for i in range(40):
        weights = {0.0: 1.0, 1.0:i+1}
        clf = LogisticRegression(random_state=0, max_iter=10000,solver='lbfgs', class_weight=weights).fit(X_train, y_train)
        y_pred_LR= clf.predict(X_test)
        # print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
        # print(classification_report(y_test, y_pred))
        Accuracy_LR.append(clf.score(X_test, y_test))
        AUC_LR.append(roc_auc_score(y_test, y_pred_LR))
        disp_LR= confusion_matrix(y_test, y_pred_LR)
        false_LR.append((disp_LR[0,1]+disp_LR[1,0])/len(y_test))

        SVM = SVC(gamma='scale', class_weight=weights)
        SVM.fit(X_train,y_train)
        y_pred_SVM=SVM.predict(X_test)
        Accuracy_SVM.append(SVM.score(X_test, y_test))
        AUC_SVM.append(roc_auc_score(y_test, y_pred_SVM))
        disp_SVM= confusion_matrix(y_test, y_pred_SVM)
        false_SVM.append((disp_SVM[0,1]+disp_SVM[1,0])/len(y_test))


    print("out of loop")
    plt.figure()
    plt.plot(AUC_LR, 'b', label="AUC")
    plt.plot(Accuracy_LR, 'g', label="Accuracy")
    plt.plot(false_LR,'r',label="False rate")
    plt.legend(loc="upper right")
    plt.show()

    plt.figure()
    plt.plot(AUC_SVM, 'b', label="AUC")
    plt.plot(Accuracy_SVM, 'g', label="Accuracy")
    plt.plot(false_SVM,'r',label="False rate")
    plt.legend(loc="upper right")
    plt.show()

    # W=AUC_LR.index(max(AUC_LR))
    # print(W)
    # clf=LogisticRegression(random_state=0, max_iter=10000,solver='lbfgs', class_weight={0.0: 1.0, 1.0:W+1}).fit(X_train, y_train)
    # # print(clf.predict_proba(X_test))
    # probability=clf.predict_proba(X_test)
    # plt.figure()
    # plt.plot(probability[:,1],y_test, 'o')
    # plt.xlabel("Probability to '1'")
    # plt.yticks([0,1])
    # plt.ylabel("y test")
    # plt.show()

    W =1# AUC_SVM.index(max(AUC_SVM))
    print(W)
    SVM = SVC(gamma='scale', class_weight={0.0: 1.0, 1.0:W+1})
    SVM.fit(X_train, y_train)
    dist = SVM.decision_function(X_test)
    # w_norm = np.linalg.norm(SVM.coef_)
    # dist = y / w_norm
    plt.figure()
    plt.plot(dist, y_test, 'o')
    plt.title(f"Accuracy Of LR For The Given Dataset : {Accuracy_SVM[W]}")
    plt.xlabel("Distance from separator")
    plt.yticks([0, 1])
    plt.ylabel("y test")
    plt.show()

    # logit_roc_auc = roc_auc_score(y_test, y_pred)
    # fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    # plt.figure()
    # plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig('Log_ROC')
    # plt.show()

    print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(clf.score(X_test, y_test)))
    # return X_train,X_test, y_train,y_test

    SVM=SVC(kernel='rbf', random_state = 1)
    SVM.fit(X_train,y_train)
    SVM_y_pred = SVM.predict(X_test)
    cm = confusion_matrix(y_test, SVM_y_pred)
    accuracy = float(cm.diagonal().sum()) / len(y_test)
    print("\nAccuracy Of SVM For The Given Dataset : ", accuracy)
    print(cm)
    print("\nAccuracy Of LR For The Given Dataset : ", Accuracy_LR[W])

def XGBoost_model(X_train, X_test, y_train, y_test):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(confusion_matrix(y_test, y_pred))

def random_forest(X_train, X_test, y_train, y_test):

    rf=RandomForestRegressor(n_estimators= 2000, min_samples_split= 10, min_samples_leaf= 2, max_features="sqrt", max_depth= 10, bootstrap= True)
    rf.fit(X_train,y_train)
    y_pred=rf.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    errors=abs(y_pred-y_test)
    AUC=roc_auc_score(y_test, y_pred)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))
    # print(confusion_matrix(y_test, y_pred))
    print("error rate=", sum(errors)/len(y_test), "AUC=", AUC)
    # fpr, tpr, thing = roc_curve(y_test,y_pred)
    # plt.figure(1)
    # plt.plot(fpr, tpr, marker='.', label='Random Forest Regression')
    # plt.plot([0,1],[0,1], linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt. legend()
    # plt.title("Random forest, auc= {:.2f}".format(AUC))
    # plt.show(block=False)
    # plt.pause(1)
    # plt.close()

def get_uncompiled_model(shape):
    inputs = keras.Input(shape=(shape,), name="digits")
    x = keras.layers.Dense(32,kernel_initializer='normal', activation="relu", name="dense_1")(inputs)
    x = keras.layers.Dense(64,kernel_initializer='normal', activation="relu", name="dense_2")(x)
    x = keras.layers.Dense(128,kernel_initializer='normal', activation="relu", name="dense_3")(x)
    x = keras.layers.Dense(256,kernel_initializer='normal', activation="relu", name="dense_4")(x)
    outputs = keras.layers.Dense(1,kernel_initializer='normal', activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def binary_recall_specificity(y_true, y_pred, recall_weight, spec_weight):
    # TN, FP, FN, TP = confusion_matrix(y_true,y_pred)

    y_pred = tf.dtypes.cast(tf.where(y_pred > 0, 1, 0), tf.float32)
    # Converted as Keras Tensors
    TN = tf.math.logical_and(K.eval(y_true) == 0, K.eval(y_pred) == 0)
    TP = tf.math.logical_and(K.eval(y_true) == 1, K.eval(y_pred) == 1)

    FP = tf.math.logical_and(K.eval(y_true) == 0, K.eval(y_pred) == 1)
    FN = tf.math.logical_and(K.eval(y_true) == 1, K.eval(y_pred) == 0)

    specificity = TN / (TN + FP + K.epsilon())
    recall = TP / (TP + FN + K.epsilon())

    return 1.0 - (recall_weight*recall + spec_weight*specificity)

    # custom_loss = lambda recall, spec: binary_recall_specificity(y_test, y_pred, recall, spec)


def DNN(X_train,X_test,y_train,y_test,epochs):

    model= Sequential()
    model.add(Dense(X_train.shape[1], kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_error'])

    history=model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.3)#, callbacks=callbacks_list)

    pred = model.predict(X_test)
    print(np.sqrt(mean_squared_error(y_test, pred)))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    y_pred=np.array(model.predict(X_test)).reshape(-1)
    # errors=abs(y_pred-y_test)
    # AUC=roc_auc_score(y_test, y_pred)
    # # print("Accuracy: %.2f%%" % (accuracy * 100.0))
    # # print(confusion_matrix(y_test, y_pred))
    # print("error rate=", sum(errors)/len(y_test), "AUC=", AUC)
    # fpr, tpr, thing = roc_curve(y_test,y_pred)
    # plt.figure()
    # plt.plot(fpr, tpr, marker='.', label='RNN model')
    # plt.plot([0,1],[0,1], linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt. legend()
    # plt.title("Area under curve= {:.2f}".format(AUC))
    # plt.show()
    #
    plt.figure()
    plt.plot(y_test,y_pred,'o')
    plt.xlabel("Test Set")
    plt.ylabel("Prdiction- binary")
    plt.show()

    logit_roc_auc = roc_auc_score(np.where(y_test > 0, 1, 0), y_pred)
    fpr, tpr, thresholds = roc_curve(np.where(y_test > 0, 1, 0), y_pred)

    accuracy=[]
    specificity=[]
    sensitivity=[]
    for threshold in thresholds:
        tn, fp, fn, tp = confusion_matrix(np.where(y_test > 0, 1, 0), np.where(y_pred > threshold, 1, 0).reshape(-1)).ravel()
        accuracy_score=(tn+tp)/(tn+fp+fn+tp)
        specificity_score = tn / (tn + fp)
        sensitivity_score=tp/(tp+fn)
        accuracy.append(accuracy_score)
        specificity.append(specificity_score)
        sensitivity.append(sensitivity_score)
        # print("threshold=","{:.2f}".format(threshold),"accuracy=","{:.2f}".format(accuracy_score), "sensitivity=","{:.2f}".format(sensitivity_score),"specificity=","{:.2f}".format(specificity_score))

    plt.figure()
    plt.plot(fpr, tpr, label='AUC = %0.2f' % logit_roc_auc)
    plt.plot(sensitivity,specificity, label="sensitivity vs. specificity")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(f'DNN model\nepochs={epochs}'.format(t))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()

# def focal_loss_lgb_f1_score(y_pred, y_test):
#   y_pred = 1 / (1 + np.exp(-y_pred))
#   binary_preds = [int(p>0) for p in y_pred]
#   return 'f1', f1_score(y_test, binary_preds), True

# def custom_loss(y_true, y_pred):
#     a, g = 0.25, 1.
#     p = 1 / (1 + K.exp(-y_pred))
#     # calculate loss, using y_pred
#     loss = K.mean(-(a * y_test + (1 - a) * (1 - y_test)) * ((1 - (y_test * p + (1 - y_test) * (1 - p))) ** g) * (
#                 y_test * K.log(p) + (1 - y_test) * K.log(1 - p)))
#     return loss

def call(y_true, y_pred):
    # if not self.from_logits:
        # with tf.name_scope('Weighted_Cross_Entropy'):
            # Manually calculated the weighted cross entropy. Formula is qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x)) where z are labels, x is logits, and q is the weight.
            # Since the values passed are from sigmoid (assumably in this case) sigmoid(x) will be replaces with y_pred

    y_pred=tf.dtypes.cast(tf.where(y_pred > 0., 1, 0), tf.float32)

    # print("ypred= ", y_pred)

    TN = tf.math.logical_and(K.eval(y_true) == 0, K.eval(y_pred) == 0)
    TN=tf.reduce_sum(tf.cast(TN,tf.float32))
    TP = tf.math.logical_and(K.eval(y_true) == 1, K.eval(y_pred) == 1)
    TP=tf.reduce_sum(tf.cast(TP,tf.float32))


    FP = tf.math.logical_and(K.eval(y_true) == 0, K.eval(y_pred) == 1)
    FP=tf.reduce_sum(tf.cast(FP,tf.float32))
    FN = tf.math.logical_and(K.eval(y_true) == 1, K.eval(y_pred) == 0)
    FN=tf.reduce_sum(tf.cast(FN,tf.float32))

    # print("TN=",TN)
    # print("FP=",FP)
    # print("TP=",TP)
    # print("FN=",FN)
    specificity = K.sum(TN) / (K.sum(TN) + K.sum(FP) + K.epsilon())
    recall = K.sum(TP) / (K.sum(TP) + K.sum(FN) + K.epsilon())

    print("recall=",recall)
    print("specificity=",specificity)

    return 1.0 - (0.7* recall + 0.3 * specificity)

def DNN_regress(X_train,X_test,y_train,y_test, parameters,name,fullname):
    model = Sequential()
    model.add(Dense(X_train.shape[1], kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=parameters["learning rate"]), loss='mse', run_eagerly=True, metrics=[call, 'mae'])

    # train model
    history = model.fit(X_train, y_train, epochs=parameters["#epochs"], batch_size=32, validation_split=0.3)
    # plot metrics
    path=Savemodel_DNN(model,name)

    plt.figure()
    # pred = model.predict(X_test)
    # print(np.sqrt(mean_squared_error(y_test, pred)))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig(f'{path}/{name}-history.jpeg')

    y_pred=model.predict(X_test)
    plt.figure()
    plt.plot(y_test,y_pred,'o')
    plt.xlabel("Test Set")
    plt.ylabel("Prdiction- binary")
    # plt.show()
    plt.savefig(f'{path}/{name}-results-DNN.jpeg')

    logit_roc_auc = float("{:.2f}".format(roc_auc_score(np.where(y_test > 0, 1, 0), y_pred)))
    fpr, tpr, thresholds = roc_curve(np.where(y_test > 0, 1, 0), y_pred)

    # # export to excel
    # df = pd.DataFrame(data={'fpr': fpr, 'tpr': tpr, 'threshold':thresholds})
    # df.to_excel(f'{path}/{name}_DNN_Values.xlsx',index=False)

    accuracy=[]
    specificity=[]
    sensitivity=[]
    for threshold in thresholds:
        tn, fp, fn, tp = confusion_matrix(np.where(y_test > 0, 1, 0), np.where(y_pred > threshold, 1, 0).reshape(-1)).ravel()
        accuracy_score=(tn+tp)/(tn+fp+fn+tp)
        specificity_score = tn / (tn + fp)
        sensitivity_score=tp/(tp+fn)
        accuracy.append(accuracy_score)
        specificity.append(specificity_score)
        sensitivity.append(sensitivity_score)
        # print("threshold=","{:.2f}".format(threshold),"accuracy=","{:.2f}".format(accuracy_score), "sensitivity=","{:.2f}".format(sensitivity_score),"specificity=","{:.2f}".format(specificity_score))
    index_80=np.argwhere(np.array(sensitivity)>0.8)[0][0]
    index_65=np.argwhere(np.array(sensitivity)>0.65)[0][0]

    # export to excel
    df = pd.DataFrame(
        data={'accuracy': accuracy, 'specificity': specificity, 'sensitivity': sensitivity, 'threshold': thresholds})
    df.to_excel(f'{path}/{name}_randomForestValues.xlsx')

    plt.figure(figsize=(15,8))
    plt.subplot(1, 2, 1, aspect='equal')
    plt.plot(fpr, tpr)
    plt.text(0.4,0.5,'AUC = %0.2f' % logit_roc_auc,fontsize=14,bbox=dict(facecolor='white', alpha=0.5))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=12)
    plt.ylabel('True Positive Rate',fontsize=12)
    plt.title(f'ROC Curve')

    plt.subplot(1, 2, 2, aspect='equal')
    plt.plot(sensitivity,specificity,'c', label="_recall vs. specificity")
    # plt.plot(sensitivity[np.argmax(accuracy)],specificity[np.argmax(accuracy)],'o', label=f'sensitivity={float("{:.2f}".format(sensitivity[np.argmax(accuracy)]))}\nspecificity={float("{:.2f}".format(specificity[np.argmax(accuracy)]))}')
    # # plt.text(sensitivity[np.argmax(accuracy)]-0.1, specificity[np.argmax(accuracy)]-0.1,f'sensitivity={float("{:.2f}".format(sensitivity[np.argmax(accuracy)]))}\nspecificity={float("{:.2f}".format(specificity[np.argmax(accuracy)]))}')
    plt.plot(sensitivity[index_80], specificity[index_80],'o', label=f'sensitivity={float("{:.2f}".format(sensitivity[index_80]))}\nspecificity={float("{:.2f}".format(specificity[index_80]))}')
    # plt.text(sensitivity[index]-0.1, specificity[index]-0.1,f'sensitivity={float("{:.2f}".format(sensitivity[index]))}\nspecificity={float("{:.2f}".format(specificity[index]))}')
    plt.plot(sensitivity[index_65], specificity[index_65],'o', label=f'sensitivity={float("{:.2f}".format(sensitivity[index_65]))}\nspecificity={float("{:.2f}".format(specificity[index_65]))}')
    # plt.text(sensitivity[index]-0.1, specificity[index]-0.1,f'sensitivity={float("{:.2f}".format(sensitivity[index]))}\nspecificity={float("{:.2f}".format(specificity[index]))}')
    plt.plot([0, 1], [0, 1],'c--')
    plt.legend(loc="upper right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Sensitivity',fontsize=12)
    plt.ylabel('Specificity',fontsize=12)
    plt.title("Sensitivity vs. Specificity")
    plt.suptitle(f'{fullname}\nMax accuracy={round(max(accuracy),2)}, learning rate={parameters["learning rate"]}, epochs={parameters["#epochs"]}',fontsize=15)
    plt.savefig(f'{path}/{name}-DNN.jpeg')
    # plt.show()

    parameters["AUC"]=logit_roc_auc
    with open(f'{path}/parameters.json', 'w') as fp:
        json.dump(parameters, fp)


def Random_forest_regress(X_train,X_test,y_train,y_test,parameters,name,fullname):
    regressor = RandomForestRegressor(n_estimators=parameters["n_estimators"], random_state=43,min_samples_leaf= 2, max_features="sqrt", max_depth= 12, bootstrap= True)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    path=Savemodel_RF(regressor,name)

    plt.figure()
    plt.plot(y_test,y_pred,'o')
    plt.xlabel("Test Set")
    plt.ylabel("Prdiction- binary")
    plt.title(f'{name}- results')
    # plt.show()
    plt.savefig(f'{path}/{name}-results-RandomForest.jpeg')

    logit_roc_auc = float("{:.2f}".format(roc_auc_score(np.where(y_test > 0, 1, 0), y_pred)))
    fpr, tpr, thresholds = roc_curve(np.where(y_test > 0, 1, 0), y_pred)

    accuracy=[]
    specificity=[]
    sensitivity=[]
    for threshold in thresholds:
        tn, fp, fn, tp = confusion_matrix(np.where(y_test > 0, 1, 0), np.where(y_pred > threshold, 1, 0).reshape(-1)).ravel()
        accuracy_score=(tn+tp)/(tn+fp+fn+tp)
        specificity_score = tn / (tn + fp)
        sensitivity_score=tp/(tp+fn)
        accuracy.append(accuracy_score)
        specificity.append(specificity_score)
        sensitivity.append(sensitivity_score)

    # export to excel
    df = pd.DataFrame(data={'accuracy': accuracy, 'specificity': specificity,'sensitivity':sensitivity, 'threshold':thresholds})
    df.to_excel(f'{path}/{name}_randomForestValues.xlsx')

    index_80=np.argwhere(np.array(sensitivity)>0.8)[0][0]
    index_65=np.argwhere(np.array(sensitivity)>0.65)[0][0]

    plt.figure(figsize=(15,8))
    plt.subplot(1, 2, 1, aspect='equal')
    plt.plot(fpr, tpr)
    plt.text(0.4,0.5,'AUC = %0.2f' % logit_roc_auc,fontsize=14,bbox=dict(facecolor='white', alpha=0.5))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=12)
    plt.ylabel('True Positive Rate',fontsize=12)
    plt.title(f'ROC Curve')

    plt.subplot(1, 2, 2, aspect='equal')
    plt.plot(sensitivity,specificity, label="_recall vs. specificity")
    # plt.plot(sensitivity[np.argmax(accuracy)],specificity[np.argmax(accuracy)],'o', label=f'sensitivity={float("{:.2f}".format(sensitivity[np.argmax(accuracy)]))}\nspecificity={float("{:.2f}".format(specificity[np.argmax(accuracy)]))}')
    # # plt.text(sensitivity[np.argmax(accuracy)]-0.1, specificity[np.argmax(accuracy)]-0.1,f'sensitivity={float("{:.2f}".format(sensitivity[np.argmax(accuracy)]))}\nspecificity={float("{:.2f}".format(specificity[np.argmax(accuracy)]))}')
    plt.plot(sensitivity[index_80], specificity[index_80], 'o',
             label=f'sensitivity={float("{:.2f}".format(sensitivity[index_80]))}\nspecificity={float("{:.2f}".format(specificity[index_80]))}')
    # plt.text(sensitivity[index]-0.1, specificity[index]-0.1,f'sensitivity={float("{:.2f}".format(sensitivity[index]))}\nspecificity={float("{:.2f}".format(specificity[index]))}')
    plt.plot(sensitivity[index_65], specificity[index_65], 'o',
             label=f'sensitivity={float("{:.2f}".format(sensitivity[index_65]))}\nspecificity={float("{:.2f}".format(specificity[index_65]))}')
    # plt.text(sensitivity[index]-0.1, specificity[index]-0.1,f'sensitivity={float("{:.2f}".format(sensitivity[index]))}\nspecificity={float("{:.2f}".format(specificity[index]))}')
    plt.legend(loc="upper right")
    plt.plot([0, 1], [0, 1],'c--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Sensitivity',fontsize=12)
    plt.ylabel('Specificity',fontsize=12)
    plt.title("Sensitivity vs. Specificity")
    plt.suptitle(f'{fullname}\n #of trees={parameters["n_estimators"]},Max accuracy={float("{:.2f}".format(max(accuracy)))}',fontsize=15)
    plt.savefig(f'{path}/{name}-RandomForest.jpeg')


    parameters["AUC"]=logit_roc_auc
    with open(f'{path}/parameters.json', 'w') as fp:
        json.dump(parameters, fp)

if __name__ == '__main__':
    df=pd.read_excel("./ELK_tableFalse.xlsx")
    types=["FA_Egg","FA_Milk","FA_Peanut","FA_general","SCORAD"]




    multi_df=pd.read_excel("./multimorbidityTable_2.xlsx")
    print(multi_df.shape)
    y=multi_df["primary outcome positive (only those evaluable and within age range)"]
    y=np.where(y > 0, 1, 0)
    X=multi_df.drop(columns=["primary outcome positive (only those evaluable and within age range)"])
    test_size = 0.2
    epochs = 20
    n_estimators = 1000
    lr = 0.0001
    parametrs_DNN = {"description": "", "test_size": test_size, "#epochs": epochs, "learning rate": lr}
    parametrs_RF = {"description": "", "test_size": test_size, "n_estimators": n_estimators, "#epochs": epochs}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parametrs_DNN["test_size"], stratify=y)
    Random_forest_regress(X_train, X_test, y_train, y_test,parametrs_RF, name='Multimorbidity',fullname='Multimorbidity')
    DNN_regress(X_train, X_test, y_train, y_test, parametrs_DNN,name='Multimorbidity',fullname='Multimorbidity')

    # FA, label, name,fullname=Type('AD')
    #
    # merged_df=create_tables(run_tables_creation=False,FA=FA)
    # # print(merged_df.describe())
    # test_size = 0.2
    # epochs=100
    # n_estimators=600
    # lr = 0.0001
    # parametrs_DNN={"description":"", "test_size":test_size,"#epochs":epochs,"learning rate":lr}
    # parametrs_RF={"description":"", "test_size":test_size,"n_estimators":n_estimators,"#epochs":epochs}
    #
    # y =merged_df[label]
    # y_binary=np.where(y > 0, 1, 0)
    # # print("FA",Counter(np.where(y>0,1,0))) #FA Counter({0: 1135, 1: 260}) (16%)
    # # print("AD",Counter(np.where(y>0,1,0))) #AD Counter({0: 1018, 1: 448}) (30%)
    #
    # X = merged_df.drop(columns=[label])
    # X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=parametrs_DNN["test_size"], stratify=y_binary)
    # # SVM_logisticRedression_model(X_train, X_test, y_train, y_test)
    # # XGBoost_model(X_train, X_test, y_train, y_test)
    # Random_forest_regress(X_train, X_test, y_train, y_test,parametrs_RF, name=name,fullname=fullname) #n_estimators
    # DNN_regress(X_train, X_test, y_train, y_test, parametrs_DNN,name=name,fullname=fullname) ##epochs, learning rate
    # # DNN(X_train, X_test, np.where(y_train > 0, 1, 0), np.where(y_test > 0, 1, 0),epochs=200)

    # FA, label, name,fullname=Type('FA')
    # merged_df=create_tables(run_tables_creation=False,FA=FA)
    # test_size = 0.2
    # epochs=100
    # n_estimators=600
    # lr = 0.001
    # parametrs_DNN={"description":"", "test_size":test_size,"#epochs":epochs,"learning rate":lr}
    # parametrs_RF={"description":"", "test_size":test_size,"n_estimators":n_estimators,"#epochs":epochs}
    # y =merged_df[label]
    # y_binary=np.where(y > 0, 1, 0)
    #
    # X = merged_df.drop(columns=[label])
    # X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=test_size, stratify=y_binary)
    # Random_forest_regress(X_train, X_test, y_train, y_test,parametrs_RF, name=name,fullname=fullname)
    # DNN_regress(X_train, X_test, y_train, y_test, parametrs_DNN,name=name,fullname=fullname)
