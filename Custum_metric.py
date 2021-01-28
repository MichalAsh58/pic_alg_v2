import keras
import tensorflow as tf
import keras.backend as K

class ConfusionMatrixMetric(tf.keras.metrics.Metric):

    def __init__(self, name='confusionMatrixMetric'):
        super(ConfusionMatrixMetric, self).__init__( name=name)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true, y_pred))
        return self.total_cm

    def result(self):
        return self.process_confusion_matrix()

    def confusion_matrix(self, y_true, y_pred):
        """
        Make a confusion matrix
        """
        y_pred = tf.argmax(y_pred, 1)
        cm = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32, num_classes=self.num_classes)
        return cm

    def process_confusion_matrix(self):
        "returns precision, recall and f1 along with overall accuracy"
        cm = self.total_cm
        diag_part = tf.linalg.diag_part(cm)
        precision = diag_part / (tf.reduce_sum(cm, 0) + tf.constant(1e-15))
        recall = diag_part / (tf.reduce_sum(cm, 1) + tf.constant(1e-15))
        f1 = 2 * precision * recall / (precision + recall + tf.constant(1e-15))
        return precision, recall, f1