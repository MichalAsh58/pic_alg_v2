import keras
import tensorflow as tf
import keras.backend as K

class WeightedBinaryCrossEntropy(tf.keras.metrics.Metric):
    """
    pos_weight: Scalars the effec on loss by the positive class by whatever is passed into it.
    weight: Scalars all the loss. Can be used to increase scalar of negative weight only by passing 1/weight to pos_weight.
            To affect pos_weight even more after this multiply in the other scalar you had in mind for it
    """
    def __init__(self, recall_weight, spec_weight, from_logits=False, reduction=keras.losses.Reduction.AUTO, name='weighted_binary_crossentropy'):
        super(WeightedBinaryCrossEntropy, self).__init__(reduction=reduction, name=name)
        self.recall_weight = recall_weight
        self.spec_weight = spec_weight
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
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

        # loss=1.0 - (self.recall_weight * recall + self.spec_weight * specificity))
        #
        # with tf.GradientTape() as tape:
        #     g = tape.gradient(loss,
        #

        return 1.0 - (self.recall_weight * recall + self.spec_weight * specificity)




        # x_1 = y_true * self.recall_weight * -tf.math.log(
        #     y_pred + 1e-6)  # qz * -log(sigmoid(x)) 1e-6 is added as an epsilon to stop passing a zero into the log
        # x_2 = (1 - y_true) * -tf.math.log(
        #     1 - y_pred + 1e-6)  # (1 - z) * -log(1 - sigmoid(x)). Epsilon is added to prevent passing a zero into the log
        # print("lulu",type(tf.add(x_1,x_2)))
        # return tf.add(x_1,x_2) * self.spec_weight  # Must be negative as it is maximized when passed to optimizers
        #
        # # # Use built in function
        # # return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.pos_weight) * self.weight

