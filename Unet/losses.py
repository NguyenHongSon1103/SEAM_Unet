import tensorflow as tf

beta = 0.25
alpha = 0.25
gamma = 2
epsilon = 1e-5
smooth = 1


class Semantic_loss_functions(object):
    def __init__(self):
        print ("semantic loss functions initialized")
        self.epsilon = 1e-7

#     def dice_coef(self, y_true, y_pred):
#         y_true_f = tf.keras.layers.Flatten()(y_true)
#         y_pred_f = tf.keras.layers.Flatten()(y_pred)
#         intersection = tf.reduce_sum(y_true_f * y_pred_f)
#         return (2. * intersection + self.epsilon) / (
#                     tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + self.epsilon)
    
    def dice_coef(self, y_true, y_pred):
        '''
        Sørensen–Dice coefficient for 2-d samples.

        Input
        ----------
            y_true, y_pred: predicted outputs and targets.
            const: a constant that smooths the loss gradient and reduces numerical instabilities.

        '''

        # flatten 2-d tensors
        y_true_pos = tf.reshape(y_true, [-1])
        y_pred_pos = tf.reshape(y_pred, [-1])

        # get true pos (TP), false neg (FN), false pos (FP).
        true_pos  = tf.reduce_sum(y_true_pos * y_pred_pos)
        false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))
        false_pos = tf.reduce_sum((1-y_true_pos) * y_pred_pos)

        # 2TP/(2TP+FP+FN) == 2TP/()
        coef_val = (2.0 * true_pos + self.epsilon)/(2.0 * true_pos + false_pos + false_neg)

        return coef_val
    
    def dice_loss(self, y_true, y_pred):
        '''
        Sørensen–Dice Loss.

        dice(y_true, y_pred, const=K.epsilon())

        Input
        ----------
            const: a constant that smooths the loss gradient and reduces numerical instabilities.

        '''
        # tf tensor casting
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        # <--- squeeze-out length-1 dimensions.
        y_pred = tf.squeeze(y_pred)
        y_true = tf.squeeze(y_true)

        loss_val = 1 - self.dice_coef(y_true, y_pred)

        return loss_val

    def sensitivity(self, y_true, y_pred):
        true_positives = tf.reduce_sum(tf.math.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
        possible_positives = tf.reduce_sum(tf.math.round(tf.clip_by_value(y_true, 0, 1)))
        return true_positives / (possible_positives + self.epsilon)

    def specificity(self, y_true, y_pred):
        true_negatives = tf.reduce_sum(
            tf.math.round(tf.clip_by_value((1 - y_true) * (1 - y_pred), 0, 1)))
        possible_negatives = tf.reduce_sum(tf.math.round(tf.clip_by_value(1 - y_true, 0, 1)))
        return true_negatives / (possible_negatives + self.epsilon)

    def convert_to_logits(self, y_pred):
        y_pred = tf.clip_by_value(y_pred, self.epsilon,
                                  1 - self.epsilon)
        return tf.math.log(y_pred / (1 - y_pred))

    def weighted_cross_entropyloss(self, y_true, y_pred):
        y_pred = self.convert_to_logits(y_pred)
        pos_weight = beta / (1 - beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred,
                                                        targets=y_true,
                                                        pos_weight=pos_weight)
        return tf.reduce_mean(loss)

    def focal_loss_with_logits(self, logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(
            -logits)) * (weight_a + weight_b) + logits * weight_b

    def focal_loss(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, self.epsilon,
                                  1 - self.epsilon)
        logits = tf.math.log(y_pred / (1 - y_pred))

        loss = self.focal_loss_with_logits(logits=logits, targets=y_true,
                                      alpha=alpha, gamma=gamma, y_pred=y_pred)

        return tf.reduce_mean(loss)

    def depth_softmax(self, matrix):
        sigmoid = lambda x: 1 / (1 + tf.math.exp(-x))
        sigmoided_matrix = sigmoid(matrix)
        softmax_matrix = sigmoided_matrix / tf.reduce_sum(sigmoided_matrix, axis=0)
        return softmax_matrix

    def bce_dice_loss(self, y_true, y_pred):
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + \
               self.dice_loss(y_true, y_pred)
        return loss / 2.0

    def confusion(self, y_true, y_pred):
        smooth = 1
        y_pred_pos = tf.clip_by_value(y_pred, 0, 1)
        y_pred_neg = 1 - y_pred_pos
        y_pos = tf.clip_by_value(y_true, 0, 1)
        y_neg = 1 - y_pos
        tp = tf.reduce_sum(y_pos * y_pred_pos)
        fp = tf.reduce_sum(y_neg * y_pred_pos)
        fn = tf.reduce_sum(y_pos * y_pred_neg)
        prec = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        return prec, recall

    def true_positive(self, y_true, y_pred):
        smooth = 1
        y_pred_pos = tf.math.round(tf.clip_by_value(y_pred, 0, 1))
        y_pos = tf.math.round(tf.clip_by_value(y_true, 0, 1))
        tp = (tf.reduce_sum(y_pos * y_pred_pos) + smooth) / (tf.reduce_sum(y_pos) + smooth)
        return tp

    def true_negative(self, y_true, y_pred):
        smooth = 1
        y_pred_pos = tf.math.round(tf.clip_by_value(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos
        y_pos = tf.math.round(tf.clip_by_value(y_true, 0, 1))
        y_neg = 1 - y_pos
        tn = (tf.reduce_sum(y_neg * y_pred_neg) + smooth) / (tf.reduce_sum(y_neg) + smooth)
        return tn

    def tversky_index(self, y_true, y_pred):
        y_true_pos = tf.keras.layers.Flatten()(y_true)
        y_pred_pos = tf.keras.layers.Flatten()(y_pred)
        true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
        false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
        false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.7
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (
                    1 - alpha) * false_pos + smooth)

    def tversky_loss(self, y_true, y_pred):
        return 1 - self.tversky_index(y_true, y_pred)

    def focal_tversky(self, y_true, y_pred):
        pt_1 = self.tversky_index(y_true, y_pred)
        gamma = 0.75
        return tf.math.pow((1 - pt_1), gamma)

    def log_cosh_dice_loss(self, y_true, y_pred):
        x = self.dice_loss(y_true, y_pred)
        return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)

if __name__ == '__main__':
    import numpy as np
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    ytrue = np.random.random((5, 64, 320, 1))
    ypred = np.random.random((5, 64, 320))
    loss_fn = Semantic_loss_functions().dice_coef
    loss = loss_fn(ytrue, ypred)
    print(loss)
