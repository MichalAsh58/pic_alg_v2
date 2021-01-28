import keras
import tensorflow as tf
import keras.backend as K

def call(y_true, y_pred):
    # if not self.from_logits:
        # with tf.name_scope('Weighted_Cross_Entropy'):
            # Manually calculated the weighted cross entropy. Formula is qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x)) where z are labels, x is logits, and q is the weight.
            # Since the values passed are from sigmoid (assumably in this case) sigmoid(x) will be replaces with y_pred

    y_pred=tf.dtypes.cast(tf.where(y_pred > 0., 1, 0), tf.float32)

    print("ypred= ", y_pred)

    TN = tf.math.logical_and(K.eval(y_true) == 0, K.eval(y_pred) == 0)
    TN=tf.reduce_sum(tf.cast(TN,tf.float32))
    TP = tf.math.logical_and(K.eval(y_true) == 1, K.eval(y_pred) == 1)
    TP=tf.reduce_sum(tf.cast(TP,tf.float32))


    FP = tf.math.logical_and(K.eval(y_true) == 0, K.eval(y_pred) == 1)
    FP=tf.reduce_sum(tf.cast(FP,tf.float32))
    FN = tf.math.logical_and(K.eval(y_true) == 1, K.eval(y_pred) == 0)
    FN=tf.reduce_sum(tf.cast(FN,tf.float32))


    print("TN=",TN)
    print("FP=",FP)
    print("TP=",TP)
    print("FN=",FN)
    specificity = K.sum(TN) / (K.sum(TN) + K.sum(FP) + K.epsilon())
    recall = K.sum(TP) / (K.sum(TP) + K.sum(FN) + K.epsilon())

    print("recall=",recall)
    print("specificity=",specificity)

    return 1.0 - (0.7* recall + 0.3 * specificity)

if __name__ == '__main__':
#regression
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