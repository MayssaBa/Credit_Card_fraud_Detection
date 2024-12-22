import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sympy.physics.quantum.matrixutils import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


#train dataset:
df_train=pd.read_csv("fraudTrain.csv")

#test dataset:
df_test=pd.read_csv("fraudTest.csv")

#verifier si le dataset est desequilibre:
print("\nAvant equilibrer les données: \n")
#************pour le dataframe train *****:
non_fraud_train=df_train[df_train["is_fraud"]==0]
fraud_train=df_train[df_train["is_fraud"]==1]
print("nb de cas non_fraud dans train=",non_fraud_train.shape[0])
print("nb de cas fraud dans train=",fraud_train.shape[0])

#***********pour le dataframe test *****:
non_fraud_test=df_test[df_test["is_fraud"]==0]
fraud_test=df_test[df_test["is_fraud"]==1]
print("\nnb de cas non_fraud dans test=",non_fraud_test.shape[0])
print("nb de cas fraud dans test=",fraud_test.shape[0])

#visualisation pour les classes des dataframes:
plt.figure(figsize=(14, 6))

# Histogramme pour le dataframe Train
plt.subplot(1, 2, 1)
sns.histplot(df_train['is_fraud'], bins=2, kde=False, palette='viridis', color='blue')
plt.title("Distribution des classes dans le dataset Train", fontsize=14)
plt.xlabel("Classes", fontsize=12)
plt.ylabel("Nombre de cas", fontsize=12)
plt.xticks([0, 1], labels=["Non-Fraud", "Fraud"])

# Histogramme pour le dataframe Test
plt.subplot(1, 2, 2)
sns.histplot(df_test['is_fraud'], bins=2, kde=False, palette='viridis', color='green')
plt.title("Distribution des classes dans le dataset Test", fontsize=14)
plt.xlabel("Classes", fontsize=12)
plt.ylabel("Nombre de cas", fontsize=12)
plt.xticks([0, 1], labels=["Non-Fraud", "Fraud"])

plt.show()


# *******Equilibré les DataFrame:
print("\nAprés equilibrer les données: \n")
#dimunier le nbre des cas Non Fraud:
non_fraud_sample_train=non_fraud_train.sample(n=7506,random_state=42)
non_fraud_sample_test=non_fraud_test.sample(n=2145,random_state=42)

df_train_equilibre=pd.concat([non_fraud_sample_train,fraud_train])
df_test_equilibre=pd.concat([non_fraud_sample_test,fraud_test])

print("distribution de classe dans train: ",df_train_equilibre['is_fraud'].value_counts())
print("\ndistribution de classe dans test:",df_test_equilibre['is_fraud'].value_counts())

#visualisation pour les classes de dataframes equilibre:
plt.figure(figsize=(14, 6))

# Histogramme pour le dataframe Train
plt.subplot(1, 2, 1)
sns.histplot(df_train_equilibre['is_fraud'], bins=2, kde=False, palette='viridis', color='blue')
plt.title("Distribution des classes dans le dataset Train equilibré", fontsize=14)
plt.xlabel("Classes", fontsize=12)
plt.ylabel("Nombre de cas", fontsize=12)
plt.xticks([0, 1], labels=["Non-Fraud", "Fraud"])

# Histogramme pour le dataframe Test
plt.subplot(1, 2, 2)
sns.histplot(df_test_equilibre['is_fraud'], bins=2, kde=False, palette='viridis', color='green')
plt.title("Distribution des classes dans le dataset Test equilibré", fontsize=14)
plt.xlabel("Classes", fontsize=12)
plt.ylabel("Nombre de cas", fontsize=12)
plt.xticks([0, 1], labels=["Non-Fraud", "Fraud"])

plt.show()

#split data :
#****train:
X_train=df_train_equilibre.drop(columns=["is_fraud"])
y_train=df_train_equilibre['is_fraud']
#****test :
X_test=df_test_equilibre.drop(columns=["is_fraud"])
y_test=df_test_equilibre['is_fraud']

#**************************************************transformer String Data ==> Data numeriques **** TRAINING DATA *****:
#faire ces modifs sur des copies :
X_train_num=X_train.copy()

# trans_date_trans_time feature:
X_train_num['trans_date_trans_time'] = pd.to_datetime(X_train_num['trans_date_trans_time'])
X_train_num['hour'] = X_train_num['trans_date_trans_time'].dt.hour
X_train_num['day'] = X_train_num['trans_date_trans_time'].dt.day
X_train_num['month'] = X_train_num['trans_date_trans_time'].dt.month
X_train_num.drop(columns=['trans_date_trans_time'], inplace=True)

#date X_train_num of birth==>change it to age:
X_train_num['dob'] = pd.to_datetime(X_train_num['dob'])
X_train_num['age'] = (pd.to_datetime('today') - X_train_num['dob']).dt.days // 365
X_train_num.drop(columns=['dob'], inplace=True)

#drop first and last name, merchant ,trans_num features:
X_train_num.drop(columns=['merchant','first', 'last','trans_num'], inplace=True)

#************************************************** tansformer String Data ==> Data numeriques *** TEST DATA ****:
X_test_num=X_test.copy()

#trans_date_trans_time feature:
X_test_num['trans_date_trans_time'] = pd.to_datetime(X_test_num['trans_date_trans_time'])
X_test_num['hour'] = X_test_num['trans_date_trans_time'].dt.hour
X_test_num['day'] = X_test_num['trans_date_trans_time'].dt.day
X_test_num['month'] = X_test_num['trans_date_trans_time'].dt.month
X_test_num.drop(columns=['trans_date_trans_time'], inplace=True, errors='ignore')

#date of birth==> change it to age:
X_test_num['dob'] = pd.to_datetime(X_test_num['dob'])
X_test_num['age'] = (pd.to_datetime('today') - X_test_num['dob']).dt.days // 365
X_test_num.drop(columns=['dob'], inplace=True)

#drop first and last name, merchant and trans_num features:
X_test_num.drop(columns=['merchant','first', 'last','trans_num'], inplace=True,errors='ignore')

#**************************************************TEST et TRAIN data****:
#category , gender , street, city, state, zip, job features in TEST et TRAIN:
encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
categorical_cols = [ 'category', 'gender', 'street', 'city', 'state', 'zip', 'job']
X_train_encoded_df = pd.DataFrame()
X_test_encoded_df = pd.DataFrame()
for col in categorical_cols:
    # Fit and transform on training data
    encoded_data_train = encoder.fit_transform(X_train_num[[col]])
    # Get feature names after encoding
    feature_names = encoder.get_feature_names_out([col])
    # Create a DataFrame for encoded training data
    encoded_df_train = pd.DataFrame(encoded_data_train, columns=feature_names, index=X_train_num.index)
    X_train_encoded_df = pd.concat([X_train_encoded_df, encoded_df_train], axis=1)

    # Transform test data using the same encoder
    encoded_data_test = encoder.transform(X_test_num[[col]])
    # Create a DataFrame for encoded test data
    encoded_df_test = pd.DataFrame(encoded_data_test, columns=feature_names, index=X_test_num.index)
    X_test_encoded_df = pd.concat([X_test_encoded_df, encoded_df_test], axis=1)

# Drop original categorical columns and concatenate encoded features
X_train_num = X_train_num.drop(columns=categorical_cols)
X_train_num = pd.concat([X_train_num, X_train_encoded_df], axis=1)

X_test_num = X_test_num.drop(columns=categorical_cols)
X_test_num = pd.concat([X_test_num, X_test_encoded_df], axis=1)


"""for col in categorical_cols:
    X_train_encoded = encoder.fit_transform(X_train_num[col].values.reshape(-1, 1))
    X_test_encoded = encoder.transform(X_test_num[col].values.reshape(-1, 1))
    X_train_num[col] = X_train_encoded
    X_test_num[col] = X_test_encoded"""

#***************************************************************************************
scaler = StandardScaler()
X_train_sc= scaler.fit_transform(X_train_num)
X_test_sc= scaler.transform(X_test_num)

#***************************************************************************************Models:
#SVM:
model=svm.SVC(kernel='rbf')
model.fit(X_train_sc,y_train)

y_pred_svm=model.predict(X_test_sc)
matrice_conf_svm=confusion_matrix(y_test,y_pred_svm)
rappel_svm=recall_score(y_test,y_pred_svm)
precision_svm=precision_score(y_test,y_pred_svm)
f1_svm=f1_score(y_test,y_pred_svm)
print("matrice de confusion (svm):\n",matrice_conf_svm)
print("Rappel (svm):",rappel_svm)
print("Precision (svm) : ",precision_svm)
print("F1 score (svm): ",f1_svm)

#visualisation du matrice
plt.figure(figsize=(6, 5))
sns.heatmap(matrice_conf_svm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])

plt.title('Matrice de confusion (SVM)')
plt.xlabel('Prédictions')
plt.ylabel('Véritables valeurs')
plt.show()

#Logistic Regression:
model_lr=LogisticRegression()
model_lr.fit(X_train_sc,y_train)

y_pred_lr=model_lr.predict(X_test_sc)

matrice_conf_log_reg=confusion_matrix(y_test,y_pred_lr)
rappel_log_reg=recall_score(y_test,y_pred_lr)
precision_log_reg=precision_score(y_test,y_pred_lr)
f1_log_reg=f1_score(y_test,y_pred_lr)
print("matrice de confusion (logistic regression):\n",matrice_conf_log_reg)
print("Rappel (logistic regression):",rappel_log_reg)
print("Precision (logistic regression): ",precision_log_reg)
print("F1 score (logistic regression): ",f1_log_reg)

#visualisation du matrice
plt.figure(figsize=(6, 5))
sns.heatmap(matrice_conf_log_reg, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])

plt.title('Matrice de confusion (Logistic Regression)')
plt.xlabel('Prédictions')
plt.ylabel('Véritables valeurs')
plt.show()


#random forest
clf = RandomForestClassifier()
clf.fit(X_train_sc,y_train)
y_pred_rf = clf.predict(X_test_sc)
clf.score(X_test_sc,y_test)
print(classification_report(y_test,y_pred_rf))
#matrice de conf
matrice_conf_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(matrice_conf_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Matrice de confusion (Random Forest)')
plt.xlabel('Prédictions')
plt.ylabel('Véritables valeurs')
plt.show()

rappel_rf=recall_score(y_test,y_pred_rf)
precision_rf=precision_score(y_test,y_pred_rf)
f1_rf=f1_score(y_test,y_pred_rf)
print("matrice de confusion (Random Forest):\n",matrice_conf_rf)
print("Rappel (Random Forest):",rappel_rf)
print("Precision (Random Forest): ",precision_rf)
print("F1 score (Random Forest): ",f1_rf)
#knn
knn_model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_model.fit(X_train_sc, y_train)
y_pred_knn = knn_model.predict(X_test_sc)
knn_model.score(X_test_sc,y_test)
print(classification_report(y_test,y_pred_knn))
#matrice con knn
matrice_conf_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(matrice_conf_knn, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Matrice de confusion (KNN)')
plt.xlabel('Prédictions')
plt.ylabel('Véritables valeurs')
plt.show()


rappel_knn=recall_score(y_test,y_pred_knn)
precision_knn=precision_score(y_test,y_pred_knn)
f1_knn=f1_score(y_test,y_pred_rf)
print("matrice de confusion (KNN):\n",matrice_conf_knn)
print("Rappel (KNN):",rappel_knn)
print("Precision (KNN): ",precision_knn)
print("F1 score (KNN): ",f1_knn)
#visualisation de comparaison entre svm model et logistic regression model:
f1_scores = pd.DataFrame({
    'Model': ['SVM', 'Logistic Regression',"Random Forest","KNN"],
    'F1 Score': [f1_svm, f1_log_reg,f1_rf,f1_knn]
})
plt.figure(figsize=(6, 5))
sns.barplot(x='F1 Score', y='Model', data=f1_scores, palette='Blues')
plt.title('Comparaison entre les modeles : F1 Score ')
plt.xlabel('F1 Score')
plt.ylabel('Model')
plt.show()

