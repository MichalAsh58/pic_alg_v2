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
from main import Type,create_tables
import tensorflow_addons as tfa
import gym
from collections import Counter

from Custum_metric import ConfusionMatrixMetric
from Custom_loss import WeightedBinaryCrossEntropy
from tables_creation import EAT_Table,LEAP_db,Katz_db,CARE_data


def Random_forest_regress(X_train,X_test,y_train,y_test,CARE_df,n_estimators,name):
    regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=43,min_samples_leaf= 2, max_features="sqrt", max_depth= 12, bootstrap= True)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    logit_roc_auc = roc_auc_score(np.where(y_test > 0, 1, 0), y_pred)


    plt.figure()
    plt.plot(y_test,y_pred,'o')
    plt.xlabel("Test Set")
    plt.ylabel("Prdiction- binary")
    plt.title(f'{name}- results')
    plt.show()
    # plt.savefig(f'/home/michal/MYOR Dropbox/R&D/Allergies Product Development/Prediction/Algorithm_Beta/18_01_2021_CARE_results/{name}-results-RandomForest.jpeg')

    logit_roc_auc = roc_auc_score(np.where(y_test > 0, 1, 0), y_pred)
    fpr, tpr, thresholds = roc_curve(np.where(y_test > 0, 1, 0), y_pred)

    # # export to excel
    # df = pd.DataFrame(data={'fpr': fpr, 'tpr': tpr, 'threshold':thresholds})
    # df.to_excel(f'/home/michal/MYOR Dropbox/R&D/Allergies Product Development/Prediction/Algorithm_Beta/18_01_2021_CARE_results/{name}_randomForestValues.xlsx',index=False)

    CARE_predict=regressor.predict(CARE_df)

    accuracy=[]
    specificity=[]
    sensitivity=[]
    pred_yes=[]
    percent_yes=[]
    for threshold in thresholds:
        tn, fp, fn, tp = confusion_matrix(np.where(y_test > 0, 1, 0), np.where(y_pred > threshold, 1, 0).reshape(-1)).ravel()
        accuracy_score=(tn+tp)/(tn+fp+fn+tp)
        specificity_score = tn / (tn + fp)
        sensitivity_score=tp/(tp+fn)
        accuracy.append(accuracy_score)
        specificity.append(specificity_score)
        sensitivity.append(sensitivity_score)
        pred_yes.append(sum(np.where(CARE_predict > threshold, 1, 0)))
        percent_yes.append((sum(np.where(CARE_predict > threshold, 1, 0)))/len(CARE_predict))

    df = pd.DataFrame(data={'thresholds': thresholds, 'specificity': specificity, 'sensitivity': sensitivity,'pred_yes':pred_yes,'percent_yes':percent_yes})
    df.to_excel(f'/home/michal/MYOR Dropbox/R&D/Allergies Product Development/Prediction/Algorithm_Beta/18_01_2021_CARE_results/{name}_CARE_values_forest_1.xlsx',index=False)


    index_80=np.argwhere(np.array(sensitivity)>0.8)[0][0]
    index_65=np.argwhere(np.array(sensitivity)>0.65)[0][0]

    plt.figure()
    plt.plot(fpr, tpr, label='AUC = %0.2f' % logit_roc_auc)
    plt.plot(sensitivity,specificity, label="recall vs. specificity")
    plt.plot(sensitivity[np.argmax(accuracy)],specificity[np.argmax(accuracy)],'o')
    plt.text(sensitivity[np.argmax(accuracy)]-0.1, specificity[np.argmax(accuracy)]-0.1,f'Threshold for max\naccuracy={round(thresholds[np.argmax(accuracy)],2)}')
    plt.plot(sensitivity[index_80], specificity[index_80],'o')
    plt.text(sensitivity[index_80]-0.1, specificity[index_80]-0.1,f'recall={round(sensitivity[index_80],2)}, spec={round(specificity[index_80],2)}\n Threshold={round(thresholds[index_80],2)}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Random Forest- {name}\n #of trees={n_estimators},Max accuracy={round(max(accuracy),2)}')
    plt.legend(loc="lower right")
    plt.show()


def DNN_regress(X_train,X_test,y_train,y_test,CARE_df, epochs,lr,name):
    model = Sequential()
    model.add(Dense(X_train.shape[1], kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse', run_eagerly=True, metrics=['mae'])

    # model.compile(loss=loss, optimizer='adam', metrics=['mse', 'mae'])
    # model.compile(
    #     loss='binary_crossentropy',
    #     optimizer=tf.keras.optimizers.RMSprop(0.001),
    #     metrics=[sensitivity(), specificity]
    # )

    # train model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.3)
    # plot metrics

    plt.figure()
    # pred = model.predict(X_test)
    # print(np.sqrt(mean_squared_error(y_test, pred)))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    y_pred=model.predict(X_test)
    y_pred_bool=np.where(y_pred > 0, 1, 0)
    plt.figure()
    plt.plot(y_test,y_pred,'o')
    plt.xlabel("Test Set")
    plt.ylabel("Prdiction- binary")
    # plt.show()
    # plt.savefig(f'/home/michal/MYOR Dropbox/R&D/Allergies Product Development/Prediction/Algorithm_Beta/18_01_2021_CARE_results/{name}-results-DNN.jpeg')

    logit_roc_auc = roc_auc_score(np.where(y_test > 0, 1, 0), y_pred)
    fpr, tpr, thresholds = roc_curve(np.where(y_test > 0, 1, 0), y_pred)

    # # export to excel
    # df = pd.DataFrame(data={'fpr': fpr, 'tpr': tpr, 'threshold':thresholds})
    # df.to_excel(f'/home/michal/MYOR Dropbox/R&D/Allergies Product Development/Prediction/Algorithm_Beta/18_01_2021_CARE_results/{name}_DNN_Values.xlsx',index=False)

    CARE_predict=model.predict(CARE_df)

    accuracy=[]
    specificity=[]
    sensitivity=[]
    pred_yes=[]
    percent_yes=[]
    for threshold in thresholds:
        tn, fp, fn, tp = confusion_matrix(np.where(y_test > 0, 1, 0), np.where(y_pred > threshold, 1, 0).reshape(-1)).ravel()
        accuracy_score=(tn+tp)/(tn+fp+fn+tp)
        specificity_score = tn / (tn + fp)
        sensitivity_score=tp/(tp+fn)
        accuracy.append(accuracy_score)
        specificity.append(specificity_score)
        sensitivity.append(sensitivity_score)
        pred_yes.append(sum(np.where(CARE_predict > threshold, 1, 0))[0])
        percent_yes.append((sum(np.where(CARE_predict > threshold, 1, 0))[0])/len(CARE_predict))

    df = pd.DataFrame(data={'thresholds': thresholds, 'specificity': specificity, 'sensitivity': sensitivity,'pred_yes':pred_yes,'percent_yes':percent_yes})
    df.to_excel(f'/home/michal/MYOR Dropbox/R&D/Allergies Product Development/Prediction/Algorithm_Beta/18_01_2021_CARE_results/{name}_CARE_values_DNN_1.xlsx',index=False)

    index=np.argwhere(np.array(sensitivity)>0.8)[0][0]

    plt.figure()
    plt.plot(fpr, tpr, label='AUC= %0.2f' % logit_roc_auc)
    plt.plot(sensitivity,specificity, label="recall vs. specificity")
    plt.plot(sensitivity[np.argmax(accuracy)],specificity[np.argmax(accuracy)],'o')
    plt.text(sensitivity[np.argmax(accuracy)]-0.1, specificity[np.argmax(accuracy)]-0.1,f'Threshold for max\naccuracy={round(thresholds[np.argmax(accuracy)],2)}')
    plt.plot(sensitivity[index], specificity[index],'o')
    plt.text(sensitivity[index]-0.1, specificity[index]-0.1,f'recall={round(sensitivity[index],2)}, spec={round(specificity[index],2)}\n Threshold={round(thresholds[index],2)}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'DNN model- {name}\nMax accuracy={round(max(accuracy),2)}, learning rate={lr}, epochs={epochs}')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    # plt.show()
    # plt.savefig(f'/home/michal/MYOR Dropbox/R&D/Allergies Product Development/Prediction/Algorithm_Beta/18_01_2021_CARE_results/{name}-statistics-DNN.jpeg')

    # plt.show()
    # plt.savefig(f'/home/michal/MYOR Dropbox/R&D/Allergies Product Development/Prediction/Algorithm_Beta/18_01_2021_CARE_results/{name}-statistics-randomForest.jpeg')
    # plt.savefig('Log_ROC')


if __name__ == '__main__':
    FA, label, name=Type('AD')

    merged_df=create_tables(run_tables_creation=False,FA=FA)

    y =merged_df[label]
    X = merged_df.drop(columns=[label])
    X_train, X_test, y_train, y_test = train_test_split(X, np.where(y > 0, 1, 0), test_size=0.1, stratify=np.where(y > 0, 1, 0))

    CARE_df=CARE_data()
    print(CARE_df.shape)
    CARE_df=CARE_df[X_train.columns]

    Random_forest_regress(X_train, X_test, y_train, y_test,CARE_df,n_estimators=200, name=name)
    DNN_regress(X_train, X_test, y_train, y_test,CARE_df, epochs=200, lr=0.0001,name=name)

    FA, label, name=Type('FA')

    merged_df=create_tables(run_tables_creation=False,FA=FA)

    y =merged_df[label]
    X = merged_df.drop(columns=[label])
    X_train, X_test, y_train, y_test = train_test_split(X, np.where(y > 0, 1, 0), test_size=0.1, stratify=np.where(y > 0, 1, 0))

    CARE_df=CARE_data()
    print(CARE_df.shape)
    CARE_df=CARE_df[X_train.columns]

    Random_forest_regress(X_train, X_test, y_train, y_test,CARE_df,n_estimators=200, name=name)
    DNN_regress(X_train, X_test, y_train, y_test,CARE_df, epochs=200, lr=0.0001,name=name)