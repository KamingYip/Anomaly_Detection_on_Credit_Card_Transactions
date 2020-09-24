"""
Kaming Yip
CS677 A1 Data Science with Python
Apr. 10, 2020
Final Project
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,\
                            recall_score, precision_score, f1_score
# pip install imblearn
from imblearn.over_sampling import SMOTE
import itertools
from tabulate import tabulate
# pip install keras
# pip install tensorflow
import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam

def main():
    ########## Part 0: Data Loading and Overview ##########
    print("\n" + "#" * 25 + " Part 0: Data Loading and Overview " + "#" * 25 + "\n")
    try:
        ticker = "creditcard"
        input_dir = os.getcwd()
        input_file = os.path.join(input_dir, ticker + ".csv")
        creditcard = pd.read_csv(input_file)
        print("The dataset contains transactions made by credit cards in two days. It has {0}".\
              format(creditcard.shape[0]),
              "rows and {0} columns. There is {1} missing value(s) in the entire dataset.\n".\
              format(creditcard.shape[1], "" if creditcard.isnull().values.any() else "no"),
              "The first 5 rows in the dataset are shown as follow:\n",
              creditcard.head(),
              "\n- Feature \"Time\" contains the seconds elapsed between each transaction and the first",
              "transaction in the dataset;",
              "- Features V1, V2, ..., V28 are the numerical principal components obtained with PCA.",
              "However, due to confidentiality issues, there is no metadata about the original",
              "features provided;",
              "- Feature \"Amount\" is the transaction amount;",
              "- Feature \"Class\" is the response variable and it takes value 1 in case of fraud and",
              "0 otherwise.\n",
              sep = "\n")
        
        # The Distribution of Classes
        print("-" * 25 + " The Distribution of Classes " + "-" * 25)
        class_count = creditcard.groupby("Class")["Class"].size().to_frame(name = "Count")
        plt.figure(figsize = (5, 4))
        f_class = plt.bar(class_count.index, class_count["Count"], width = 0.6,
                          align = "center", alpha = 0.7, color = ["blue", "red"])
        plt.title("The Distribution of Classes (0 = No Fraud, 1 = Fraud)")
        plt.xlabel("Class")
        plt.xticks(class_count.index)
        plt.ylabel("Count")
        for i in f_class:
            height = i.get_height()
            plt.text(i.get_x() + i.get_width()/2.0, height,
                     "{0} ({1:.2f}%)".format(height.astype(int), height/sum(class_count["Count"])*100),
                     ha = "center", va = "bottom")
        plt.show()
        
        # The Distribution of Amount and Time for Both No-Fraud and Fraud Class
        print("\n" + "-" * 7 + " The Distribution of Amount and Time for Both No-Fraud and Fraud Class " + "-" * 7)
        normal_records = creditcard.loc[creditcard["Class"] == 0, ["Time", "Amount", "Class"]].copy()
        fraud_records = creditcard.loc[creditcard["Class"] == 1, ["Time", "Amount", "Class"]].copy()
        f_amount, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 7))
        axes[0, 0].hist(normal_records.loc[normal_records["Amount"] < 500, "Amount"], bins = 20, color = "blue")
        axes[0, 0].set_title("The Distribution of Amount for No Fraud Records")
        axes[0, 0].legend(["No Fraud\n" + "median = \${0:.2f}  sd = \${1:.2f}".\
                           format(np.median(normal_records["Amount"]), np.std(normal_records["Amount"]))],
                          loc = "upper right")
        axes[1, 0].hist(fraud_records.loc[fraud_records["Amount"] < 500, "Amount"], bins = 20, color = "red")
        axes[1, 0].set_title("The Distribution of Amount for Fraud Records")
        axes[1, 0].set_xlabel("Amount($)")
        axes[1, 0].legend(["Fraud\n" + "median = \${0:.2f}  sd = \${1:.2f}".\
                           format(np.median(fraud_records["Amount"]), np.std(fraud_records["Amount"]))],
                          loc = "upper right")
        axes[0, 1].hist(normal_records["Time"], bins = 20, color = "blue")
        axes[0, 1].set_title("The Density of Time for No Fraud Records")
        axes[0, 1].legend(["No Fraud"], loc = "upper right")
        axes[1, 1].hist(fraud_records["Time"], bins = 20, color = "red")
        axes[1, 1].set_title("The Density of Time for Fraud Records")
        axes[1, 1].set_xlabel("Time(in Seconds)")
        axes[1, 1].legend(["Fraud"], loc = "upper right")
        plt.setp((axes[0, 0], axes[1, 0]), xticks = np.arange(0, 550, 100), ylabel = "Frequency")
        plt.setp((axes[0, 1], axes[1, 1]), xticks = np.arange(0, 200000, 50000), ylabel = "Frequency")
        plt.show()
        
        # The Correlation between Features
        print("\n" + "-" * 25 + " The Correlation between Features " + "-" * 25)
        corr = creditcard.corr()
        plt.figure(figsize = (10, 7))
        plt.title("Features Correlation Plot (Pearson)")
        sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, linewidths = 0.1, cmap = "coolwarm_r")
        plt.show()
        print("As shown in the correlation plot, none of the V1 to V28 PCA components have",
              "any strong correlation to each other. What's more, some of these features",
              "are either positively or negatively correlated to the output Class.", sep = "\n")
        
    except Exception as e:
        print("Error in Part 0: {0}".format(e), end = "\n\n" + "-" * 50 + "\n\n")

    ########## Part I: Data Preprocessing ##########
    print("\n" + "#" * 30 + " Part I: Data Preprocessing " + "#" * 30 + "\n")
    try:
        print("-" * 18 + " Normalize the Feature Columns and Split the Data " + "-" * 18 + "\n")
        random_state = 677

        # normalize all columns except Class
        norm_columns = creditcard.columns.tolist()
        norm_columns = [col for col in norm_columns if col not in ["Class"]]
        creditcard_norm = creditcard.copy()
        scaler = StandardScaler().fit(creditcard_norm[norm_columns].values)
        creditcard_norm[norm_columns] = scaler.transform(creditcard_norm[norm_columns].values)
        
        # Split the data into training set and testing set
        train_df, test_df = train_test_split(creditcard_norm, test_size = 0.4, random_state = random_state)
        X_train = train_df.drop("Class", axis = 1)
        y_train = train_df["Class"]
        X_test = test_df.drop("Class", axis = 1)
        y_test = test_df["Class"]
        train_test_sum = pd.DataFrame({"Dataset": ["Training", "Testing", "Total"],
                                       "Records Num": ["{0:,}({1:.2f}%)".format(len(train_df), len(train_df)/len(creditcard_norm)*100),
                                                       "{0:,}({1:.2f}%)".format(len(test_df), len(test_df)/len(creditcard_norm)*100),
                                                       "{0:,}({1:.2f}%)".format(len(creditcard_norm), len(creditcard_norm)/len(creditcard_norm)*100)],
                                       "No Fraud Num": ["{0:,}({1:.2f}%)".format(len(train_df[train_df["Class"] == 0]), len(train_df[train_df["Class"] == 0])/len(train_df)*100),
                                                        "{0:,}({1:.2f}%)".format(len(test_df[test_df["Class"] == 0]), len(test_df[test_df["Class"] == 0])/len(test_df)*100),
                                                        "{0:,}({1:.2f}%)".format(len(creditcard_norm[creditcard_norm["Class"] == 0]),
                                                                                 len(creditcard_norm[creditcard_norm["Class"] == 0])/len(creditcard_norm)*100)],
                                       "Fraud Num": ["{0:,}({1:.2f}%)".format(len(train_df[train_df["Class"] == 1]), len(train_df[train_df["Class"] == 1])/len(train_df)*100),
                                                     "{0:,}({1:.2f}%)".format(len(test_df[test_df["Class"] == 1]), len(test_df[test_df["Class"] == 1])/len(test_df)*100),
                                                     "{0:,}({1:.2f}%)".format(len(creditcard_norm[creditcard_norm["Class"] == 1]),
                                                                              len(creditcard_norm[creditcard_norm["Class"] == 1])/len(creditcard_norm)*100)]})
        print(" " * 5 + "* Data Distribution among Training and Testing Set * ",
              tabulate(train_test_sum, headers = "keys", stralign = "right"), sep = "\n\n", end = "\n\n")
        
        # Original Data Approach: simply separates the entire dataset into training set and testing set
        print("\n" + "-" * 25 + " Original Training Data Approach " + "-" * 25 + "\n",
              "Original Training Data Approach: simply separates the entire dataset into training set",
              "and testing set.", sep = "\n")
        original_class_count = pd.DataFrame(y_train).groupby("Class")["Class"].size().to_frame(name = "Count")
        plt.figure(figsize = (5, 4))
        f_class = plt.bar(original_class_count.index, original_class_count["Count"], width = 0.6,
                          align = "center", alpha = 0.7, color = ["blue", "red"])
        plt.title("The Distribution of Original Data Classes\n(0 = No Fraud, 1 = Fraud)")
        plt.xlabel("Class")
        plt.xticks(original_class_count.index)
        plt.ylabel("Count")
        for i in f_class:
            height = i.get_height()
            plt.text(i.get_x() + i.get_width()/2.0, height,
                     "{0} ({1:.2f}%)".format(height.astype(int), height/sum(original_class_count["Count"])*100),
                     ha = "center", va = "bottom")
        plt.show()
        
        # Under-sampling Approach: deletes instances from the over-represented class, which is No Fraud (Class = 1),
        # and keeps an equal balance between the majority and minority class
        print("\n" + "-" * 30 + " Under-Sampling Approach " + "-" * 30 + "\n",
              "Under-Sampling Approach: deletes instances from the over-represented class, which is No",
              "Fraud (Class = 1), and keeps an equal balance between the majority and minority class.",
              sep = "\n")
        train_undersample = train_df.sample(frac = 1, random_state = random_state)
        fraud_df = train_undersample.loc[train_undersample["Class"] == 1]
        no_fraud_df = train_undersample.loc[train_undersample["Class"] == 0][:len(train_undersample[train_undersample["Class"] == 1])]
        undersample_df = pd.concat([fraud_df, no_fraud_df])
        undersample_df = undersample_df.sample(frac = 1, random_state = random_state)
        undersample_class_count = undersample_df.groupby("Class")["Class"].size().to_frame(name = "Count")
        
        plt.figure(figsize = (5, 4))
        f_class = plt.bar(undersample_class_count.index, undersample_class_count["Count"], width = 0.6,
                          align = "center", alpha = 0.7, color = ["blue", "red"])
        plt.title("The Distribution of Under-Sampling Data Classes\n(0 = No Fraud, 1 = Fraud)")
        plt.xlabel("Class")
        plt.xticks(undersample_class_count.index)
        plt.ylabel("Count")
        for i in f_class:
            height = i.get_height()
            plt.text(i.get_x() + i.get_width()/2.0, height,
                     "{0} ({1:.2f}%)".format(height.astype(int), height/sum(undersample_class_count["Count"])*100),
                     ha = "center", va = "bottom")
        plt.show()
        
        # Over-Sampling Approach: creates synthetic points from the minority class in order to reach
        # an equal balance between the minority and majority class
        print("\n" + "-" * 30 + " Over-Sampling Approach " + "-" * 30 + "\n",
              "Over-Sampling Approach: creates synthetic points from the minority class in order to",
              "reach an equal balance between the minority and majority class.", sep = "\n")
        sm = SMOTE(random_state = random_state)
        oversampling_X_train, oversampling_y_train = sm.fit_sample(X_train, y_train)
        oversampling_class_count = pd.DataFrame(oversampling_y_train).groupby("Class")["Class"].size().to_frame(name = "Count")
        
        plt.figure(figsize = (5, 4))
        f_class = plt.bar(oversampling_class_count.index, oversampling_class_count["Count"], width = 0.6,
                          align = "center", alpha = 0.7, color = ["blue", "red"])
        plt.title("The Distribution of Over-Sampling Data Classes\n(0 = No Fraud, 1 = Fraud)")
        plt.xlabel("Class")
        plt.xticks(oversampling_class_count.index)
        plt.ylabel("Count")
        for i in f_class:
            height = i.get_height()
            plt.text(i.get_x() + i.get_width()/2.0, height,
                     "{0} ({1:.2f}%)".format(height.astype(int), height/sum(oversampling_class_count["Count"])*100),
                     ha = "center", va = "bottom")
        plt.show()
        
    except Exception as e:
        print("Error in Part I: {0}".format(e), end = "\n\n" + "-" * 50 + "\n\n")
    
    ########## Part II: Data Modeling ##########
    print("\n" + "#" * 30 + " Part II: Data Modeling " + "#" * 30 + "\n")
    try:
        def get_recall(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall
        
        def get_precision(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
            
        def get_f1(y_true, y_pred):
            precision = get_precision(y_true, y_pred)
            recall = get_recall(y_true, y_pred)
            f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())
            return f1_score
        
        def designed_confusion_matrix(train_actual, train_pred, train_title,
                                      test_actual, test_pred, test_title, classes):
            fig = plt.figure(figsize = (8, 5))
            
            fig.add_subplot(221)
            train_cm = confusion_matrix(train_actual, train_pred)
            plt.imshow(train_cm, interpolation = "nearest", cmap = "Reds")
            plt.title(train_title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation = 45)
            plt.yticks(tick_marks, classes)
            for i, j in itertools.product(range(train_cm.shape[0]), range(train_cm.shape[1])):
                plt.text(j, i, format(train_cm[i, j], "d"), horizontalalignment = "center",
                         color = "black")
            plt.tight_layout()
            plt.ylabel("Actual Class")
            plt.xlabel("Predicted Class")
            
            fig.add_subplot(222)
            test_cm = confusion_matrix(test_actual, test_pred)
            plt.imshow(test_cm, interpolation = "nearest", cmap = "Reds")
            plt.title(test_title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation = 45)
            plt.yticks(tick_marks, classes)
            for i, j in itertools.product(range(test_cm.shape[0]), range(test_cm.shape[1])):
                plt.text(j, i, format(test_cm[i, j], "d"), horizontalalignment = "center",
                         color = "black")
            plt.tight_layout()
            plt.ylabel("Actual Class")
            plt.xlabel("Predicted Class")
            plt.show()
            
            stats_summary = pd.DataFrame(columns = ["Training Data", "Testing Data"],
                                         index = ["Accuracy", "Recall", "Precision", "F1 Score"])
            stats_set = [[train_actual, train_pred], [test_actual, test_pred]]
            for i, ele in enumerate(stats_set):
                stats_summary.iloc[:, i] = [accuracy_score(ele[0], ele[1]),
                                            recall_score(ele[0], ele[1]),
                                            precision_score(ele[0], ele[1]),
                                            f1_score(ele[0], ele[1])]
            print(tabulate(stats_summary.round(3), headers = "keys", numalign = "right"), end = "\n\n")
        
        print("-" * 35 + " Neural Networks " + "-" * 35 + "\n")
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        # Under-Sampling the Original Training Data
        print("-" * 30 + " Under-Sampling Approach " + "-" * 30)
        undersample_df_X = undersample_df.drop("Class", axis = 1)
        undersample_df_y = undersample_df["Class"]
        n_inputs = undersample_df_X.shape[1]
        undersample_Neural_model = Sequential([
                Dense(n_inputs, input_shape = (n_inputs, ), activation = "relu",
                      kernel_initializer = keras.initializers.he_uniform(seed = random_state)),
                Dense(40, activation = "relu",
                      kernel_initializer = keras.initializers.he_uniform(seed = random_state)),
                Dense(1, activation = "sigmoid",
                      kernel_initializer = keras.initializers.glorot_normal(seed = random_state))])
        undersample_Neural_model.summary()
        undersample_Neural_model.compile(Adam(lr = 0.001), loss = "binary_crossentropy",
                                         metrics = ["accuracy", get_recall, get_precision, get_f1])
        undersample_Neural_model.fit(undersample_df_X, undersample_df_y, validation_split = 0.3,
                                     batch_size = 25, epochs = 20, verbose = 2)
        
        undersample_Neural_pred_train = undersample_Neural_model.predict_classes(X_train, batch_size = 200, verbose = 0)
        undersample_Neural_pred_test = undersample_Neural_model.predict_classes(X_test, batch_size = 200, verbose = 0)
        designed_confusion_matrix(y_train, undersample_Neural_pred_train, "The Confusion Matrix of\nUnder-Sampling Approach(Training)",
                                  y_test, undersample_Neural_pred_test, "The Confusion Matrix of\nUnder-Sampling Approach(Testing)",
                                  ["No Fraud", "Fraud"])
                
        # Original Training Data
        print("-" * 25 + " Original Training Data Approach " + "-" * 25)
        n_inputs = X_train.shape[1]
        original_Neural_model = Sequential([
                Dense(n_inputs, input_shape = (n_inputs, ), activation = "relu",
                      kernel_initializer = keras.initializers.he_uniform(seed = random_state)),
                Dense(40, activation = "relu",
                      kernel_initializer = keras.initializers.he_uniform(seed = random_state)),
                Dense(1, activation = "sigmoid",
                      kernel_initializer = keras.initializers.glorot_normal(seed = random_state))])
        original_Neural_model.summary()
        original_Neural_model.compile(Adam(lr = 0.001), loss = "binary_crossentropy",
                                      metrics = ["accuracy", get_recall, get_precision, get_f1])
        original_Neural_model.fit(X_train, y_train, validation_split = 0.2,
                                  batch_size = 200, epochs = 15, shuffle = True, verbose = 2)
        original_Neural_pred_train = original_Neural_model.predict_classes(X_train, batch_size = 200, verbose = 0)
        original_Neural_pred_test = original_Neural_model.predict_classes(X_test, batch_size = 200, verbose = 0)
        designed_confusion_matrix(y_train, original_Neural_pred_train, "The Confusion Matrix of\nOriginal Approach(Training)",
                                  y_test, original_Neural_pred_test, "The Confusion Matrix of\nOriginal Approach(Testing)",
                                  ["No Fraud", "Fraud"])
        
        # Over-Sampling the Original Training Data
        print("-" * 30 + " Over-Sampling Approach " + "-" * 30)
        n_inputs = oversampling_X_train.shape[1]
        oversampling_Neural_model = Sequential([
                Dense(n_inputs, input_shape = (n_inputs, ), activation = "relu",
                      kernel_initializer = keras.initializers.he_uniform(seed = random_state)),
                Dense(40, activation = "relu",
                      kernel_initializer = keras.initializers.he_uniform(seed = random_state)),
                Dense(1, activation = "sigmoid",
                      kernel_initializer = keras.initializers.glorot_normal(seed = random_state))])
        oversampling_Neural_model.summary()
        oversampling_Neural_model.compile(Adam(lr = 0.001), loss = "binary_crossentropy",
                                          metrics = ["accuracy", get_recall, get_precision, get_f1])
        oversampling_Neural_model.fit(oversampling_X_train, oversampling_y_train, validation_split = 0.3,
                                      batch_size = 200, epochs = 15, shuffle = True, verbose = 2)
        
        oversampling_Neural_pred_train = oversampling_Neural_model.predict_classes(X_train, batch_size = 200, verbose = 0)
        oversampling_Neural_pred_test = oversampling_Neural_model.predict_classes(X_test, batch_size = 200, verbose = 0)
        designed_confusion_matrix(y_train, oversampling_Neural_pred_train, "The Confusion Matrix of\nOver-Sampling Approach(Training)",
                                  y_test, oversampling_Neural_pred_test, "The Confusion Matrix of\nOver-Sampling Approach(Testing)",
                                  ["No Fraud", "Fraud"])      
        
    except Exception as e:
        print("Error in Part II: {0}".format(e), end = "\n\n" + "-" * 50 + "\n\n")
    
    ########## Part III: Summary ##########
    print("\n" + "#" * 30 + " Part III: Summary " + "#" * 30 + "\n")
    try:
        print("Although the accuracy against both training data and testing data for each model",
              "seems acceptable, it doesn't make any sense to simply take the accuracy as the",
              "only performance metric in this case. Due to the highly imbalanced feature, even",
              "if the model predicts all records as non-fraud transactions, the accuracy will be",
              "as high as 99%. Thus, other metrics, such as Recall, Precision, and F1 Score, are",
              "introduced here in order to better evaluate the model performance.\n",
              "As illustrated from the results, all the accuracies are satisfied, but not the",
              "other metrics. To be more specific, the F1 Score is undesirable in Under-Sampling",
              "approach as 0.135 against testing data, while is around 0.8 in Original approach",
              "and 0.7 in Over-Sampling approach. The reason for this performance is that there",
              "are too many non-fraud cases that are misclassified as being fraud in the model",
              "built by the Under-Sampling approach. This could be a scenario that a cardholder",
              "gets blocked when purchasing because the bank's algorithm thought the purchase was",
              "a fraud. That's why in this case we shouldn't emphasize only in detecting fraud",
              "cases but we should also emphasie correctly categorizing non-fraud transactions.\n",
              "Another interesting finding is that, although considered a \"blackbox\" in terms",
              "of interpretability, Neural Networks work successfully in the Over-Sampling and",
              "Original approach. Most of the fraud cases have been detected as well as correctly",
              "classifying the non-fraud transactions. It suggests that Neural Network relies",
              "heavily on having sufficient data for training purposes, which is the major problem",
              "in the Under-Sampling approach. However, notice that, in Over-Sampling approach,",
              "the Neural Network model performs perfectly in training data, but slightly worse in",
              "testing data, which seems like a overfitting issue. This also could be a common issue",
              "in Neural Network, the risk of obtaining weights that lead to a local optimum rather",
              "than the global optimum, in the sense that the weights converge to values that do",
              "not provide the best fit to the training data. In this case, we can try to avoid this",
              "situation by controlling the learning rate and slowly reducing the momentum. However,",
              "there is no guarantee that the resulting weights are indeed the optimal ones.",
              sep = "\n")
        
    except Exception as e:
        print("Error in Part III: {0}".format(e), end = "\n\n" + "-" * 50 + "\n\n")

    
main()
