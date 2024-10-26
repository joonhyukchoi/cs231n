# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of about 0.39 (> 0.385) on the validation set.

# Note: you may see runtime/overflow warnings during hyper-parameter search.
# This may be caused by extreme values, and is not a bug.

# results is dictionary mapping tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (training_accuracy, validation_accuracy). The accuracy is simply the fraction
# of data points that are correctly classified.
results = {}
best_val = -1   # The highest validation accuracy that we have seen so far.
best_svm = None # The LinearSVM object that achieved the highest validation rate.

################################################################################
# TODO:                                                                        #
# Write code that chooses the best hyperparameters by tuning on the validation #
# set. For each combination of hyperparameters, train a linear SVM on the      #
# training set, compute its accuracy on the training and validation sets, and  #
# store these numbers in the results dictionary. In addition, store the best   #
# validation accuracy in best_val and the LinearSVM object that achieves this  #
# accuracy in best_svm.                                                        #
#                                                                              #
# Hint: You should use a small value for num_iters as you develop your         #
# validation code so that the SVMs don't take much time to train; once you are #
# confident that your validation code works, you should rerun the validation   #
# code with a larger value for num_iters.                                      #
################################################################################

# Provided as a reference. You may or may not want to change these hyperparameters
learning_rates = [1e-7, 5e-5]
regularization_strengths = [2.5e4, 5e4]

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
start_learning_rate, end_learning_rate = learning_rates
start_regularization_strength, end_regularization_strength = regularization_strengths

num_learning_rate_steps = 100
num_regularization_strength_steps = 100

learning_rate_step_size = (end_learning_rate - start_learning_rate) / (num_learning_rate_steps - 1)
regularization_strength_step_size = (end_regularization_strength - start_regularization_strength) / (num_regularization_strength_steps - 1)

for learning_rate in [start_learning_rate + i * learning_rate_step_size for i in range(num_learning_rate_steps)]:
  for regularization_strength in [start_regularization_strength + i * regularization_strength_step_size for i in range(num_regularization_strength_steps)]:
    svm = LinearSVM()
    loss_hist = svm.train(X_train, y_train, learning_rate, reg=regularization_strength,
                      num_iters=100, verbose=False)
    y_train_pred = svm.predict(X_train)
    y_val_pred = svm.predict(X_val)
    train_accuracy = np.mean(y_train == y_train_pred)
    val_accuracy = np.mean(y_val == y_val_pred) 
    results[(learning_rate, regularization_strength)] = (train_accuracy, val_accuracy)
    if best_val < val_accuracy:
      best_val = val_accuracy
      best_svm = svm
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)