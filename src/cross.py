import models_partc
from sklearn.model_selection import KFold, ShuffleSplit
from numpy import mean

import utils

# USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

# USE THIS RANDOM STATE FOR ALL OF YOUR CROSS VALIDATION TESTS, OR THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
    #TODO:First get the train indices and test indices for each iteration
    #Then train the classifier accordingly
    #Report the mean accuracy and mean auc of all the folds
    
    kf = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)

    accuracy_list = []
    auc_list = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        y_pred = models_partc.logistic_regression_pred(X_train, y_train, X_test)

        accuracy, auc, _, _, _ = models_partc.classification_metrics(y_pred, y_test)

        accuracy_list.append(accuracy)
        auc_list.append(auc)
    
    return mean(accuracy_list), mean(auc_list)

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
    #TODO: First get the train indices and test indices for each iteration
    #Then train the classifier accordingly
    #Report the mean accuracy and mean auc of all the iterations
    rs = ShuffleSplit(n_splits=iterNo, test_size=test_percent,
                        random_state=RANDOM_STATE)

    accuracy_list = []
    auc_list = []

    for train_index, test_index in rs.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        y_pred = models_partc.logistic_regression_pred(X_train, y_train, X_test)

        accuracy, auc, _, _, _ = models_partc.classification_metrics(y_pred, y_test)

        accuracy_list.append(accuracy)
        auc_list.append(auc)
    
    return mean(accuracy_list), mean(auc_list)


def main():
    X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
    print("Classifier: Logistic Regression__________")
    acc_k,auc_k = get_acc_auc_kfold(X,Y)
    print(("Average Accuracy in KFold CV: "+str(acc_k)))
    print(("Average AUC in KFold CV: "+str(auc_k)))
    acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
    print(("Average Accuracy in Randomised CV: "+str(acc_r)))
    print(("Average AUC in Randomised CV: "+str(auc_r)))

if __name__ == "__main__":
    main()

