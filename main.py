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
import tensorflow_addons as tfa
import gym
from collections import Counter

from Custom_loss import WeightedBinaryCrossEntropy
from tables_creation import EAT_Table,LEAP_db,Katz_db

def merge_tables(path1,path2,path3):
    df1=pd.read_excel(path1)
    df2=pd.read_excel(path2)
    df3=pd.read_excel(path3)
    # test
    DF=pd.concat([df1,df2,df3])
    print("merged table",DF.shape)
    # DF.to_excel("/home/michal/MYOR Dropbox/R&D/Partnerships/Tables/MERGED.xlsx",index=False)
    return DF

def create_tables(run_tables_creation):
    if run_tables_creation:
        path_EAT = "/home/michal/MYOR Dropbox/R&D/Partnerships/EAT/EAT_Risk_Score.xlsx"
        EAT_Table(path_EAT,merged=True)
        EAT_Table(path_EAT,merged=False)
        path_LEAP = "/home/michal/MYOR Dropbox/R&D/Partnerships/LEAP/LEAP_Data.xlsx"
        path_KATZ = "/home/michal/MYOR Dropbox/R&D/Partnerships/Katz_Study/Output dat_myor_milk_Oct    4_2020.xlsx"
        LEAP_db(path_LEAP)
        Katz_db(path_KATZ)

    merged_df=merge_tables("/home/michal/MYOR Dropbox/R&D/Partnerships/Tables/EAT.xlsx","/home/michal/MYOR Dropbox/R&D/Partnerships/Tables/LEAP.xlsx","/home/michal/MYOR Dropbox/R&D/Partnerships/Tables/KATZ.xlsx")
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
    # rf = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions={'n_estimators':  [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
    #                'max_features': ['auto', 'sqrt'],
    #                'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
    #                'min_samples_split':  [2, 5, 10],
    #                'min_samples_leaf': [1, 2, 4],
    #                'bootstrap': [True, False]}, n_iter=100, cv=3, verbose=2,
    #                                random_state=42, n_jobs=-1)
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


def DNN(X_train,X_test,y_train,y_test):

    model = get_uncompiled_model(X_train.shape[1])
    # model.compile(optimizer='adam', loss=WeightedBinaryCrossEntropy(recall_weight=0.6,spec_weight=0.3), run_eagerly=True)
    # print(y_test.shape)
    # y_train_one_hot = tf.one_hot(y_train, depth=10)
    # model.fit(X_train, y_train, batch_size=64, epochs=1)

    # model= Sequential()
    # model.add(Dense(X_train.shape[1], kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))
    # model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_error'])

    # checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
    # checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    # callbacks_list = [checkpoint]

    # pred_train = model.predict(X_train)
    # print(np.sqrt(mean_squared_error(y_train, pred_train)))

    pred = model.predict(X_test)
    print(np.sqrt(mean_squared_error(y_test, pred)))

    history=model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.3)#, callbacks=callbacks_list)

    # wights_file = 'Weights-478--18738.19831.hdf5'  # choose the best checkpoint
    # model.load_weights(wights_file)  # load it
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
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot(sensitivity,specificity, label="sensitivity vs. specificity")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
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

def DNN_regress(X_train,X_test,y_train,y_test,loss):
    model = Sequential()
    model.add(Dense(X_train.shape[1], kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss=loss, optimizer='adam', metrics=['mse', 'mae'])
    # model.compile(
    #     loss='binary_crossentropy',
    #     optimizer=tf.keras.optimizers.RMSprop(0.001),
    #     metrics=[sensitivity(), specificity]
    # )

    # train model
    history = model.fit(X_train, y_train, epochs=74, batch_size=32, validation_split=0.3)
    # plot metrics

    plt.figure()
    plt.plot(history.history['mae'], label='mae')
    if loss=='mse':
        plt.plot(history.history['mse'], label='mse')
    # plt.plot(history.history['val_loss'], label='val_loss')
    # plt.plot(history.history['mape'],label='mape')
    # plt.plot(history.history['cosine_proximity'])
    plt.legend()
    plt.show()

    y_pred=model.predict(X_test)
    y_pred_bool=np.where(y_pred > 0, 1, 0)
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
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot(sensitivity,specificity, label="sensitivity vs. specificity")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(loss)
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    print(fpr)
    print(tpr)
    print(thresholds)

def Random_forest_regress(X_train,X_test,y_train,y_test):
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    MAE=mean_absolute_error(y_test, y_pred)
    MSW=mean_squared_error(y_test, y_pred)
    RMSE=mean_squared_error(y_test, y_pred)

    y_pred_bool=np.where(y_pred > 0, 1, 0)
    plt.figure()
    plt.plot(y_test,y_pred,'o')
    plt.xlabel("Test Set")
    plt.ylabel("Prdiction- binary")
    plt.show()

    logit_roc_auc = roc_auc_score(np.where(y_test > 0, 1, 0), y_pred)
    fpr, tpr, thresholds = roc_curve(np.where(y_test > 0, 1, 0), y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()



if __name__ == '__main__':
    food_allegy_label="primary outcome positive (only those evaluable and within age range)"
    atopic_label="SCORAD"
    merged_df=create_tables(run_tables_creation=False)
    # EAT_table=pd.read_excel("/home/michal/MYOR Dropbox/R&D/Partnerships/Tables/EAT_only.xlsx")
    print(merged_df.describe())

# #regression
#     y = merged_df[atopic_label]
#     X = merged_df.drop(columns=[atopic_label])
#     # print(Counter(y))
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=np.where(y > 0, 1, 0))
#
#     # SVM_logisticRedression_model(X_train, X_test, y_train, y_test)
#     # XGBoost_model(X_train, X_test, y_train, y_test)
#     # random_forest(X_train, X_test, y_train, y_test)
#     focal_loss_eval0 = lambda x, y: focal_loss_lgb_eval_error(x, y, alpha=0., gamma=1.)
#     focal_loss_eval1 = lambda x, y: focal_loss_lgb_eval_error(x, y, alpha=0.25, gamma=1.)
#     focal_loss_eval2 = lambda x, y: focal_loss_lgb_eval_error(x, y, alpha=0.5, gamma=1.)
#     focal_loss_eval3 = lambda x, y: focal_loss_lgb_eval_error(x, y, alpha=1., gamma=1.)
#     focal_loss_eval4 = lambda x, y: focal_loss_lgb_eval_error(x, y, alpha=2., gamma=1.)
#
#     loss_method=['mae','mse',tf.keras.losses.LogCosh(),tf.keras.losses.CosineSimilarity(axis=1),tf.keras.losses.Huber()]#focal_loss_eval0,focal_loss_eval1,focal_loss_eval2,focal_loss_eval3,focal_loss_eval4]
#     for loss in loss_method:
#         DNN_regress(X_train, X_test, y_train, y_test,loss)
#     # Random_forest_regress(X_train, X_test, y_train, y_test)

#binary
    y =merged_df[atopic_label]
    X = merged_df.drop(columns=[atopic_label])
    # print(Counter(y))
    X_train, X_test, y_train, y_test = train_test_split(X, np.where(y > 0, 1, 0), test_size=0.2, stratify=np.where(y > 0, 1, 0))
    # SVM_logisticRedression_model(X_train, X_test, y_train, y_test)
    # XGBoost_model(X_train, X_test, y_train, y_test)
    # random_forest(X_train, X_test, y_train, y_test)
    DNN_regress(X_train, X_test, np.where(y_train > 0, 1, 0), np.where(y_test > 0, 1, 0), 'mse')
    # DNN(X_train, X_test, np.where(y_train > 0, 1, 0), np.where(y_test > 0, 1, 0))
