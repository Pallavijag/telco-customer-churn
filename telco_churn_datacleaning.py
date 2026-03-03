import pandas as pd
import numpy as np

df=pd.read_excel("Telco_customer_churn.xlsx")

'''    print(df.head())
print(df.shape) 
print(df.columns)
print(df.info())    '''


#current shape of dataset is (7043 rows and 33 columns)


#CHECKING AND REMOVING NULL VALUES

print((df.isnull().sum()/len(df))*100)         
df=df.drop('Churn Reason', axis=1)       
print(df.shape)
print(df.dtypes)   


#shape of dataset now is (32 columns)


#CHECKING AND CORRECTING DATATYPES

#checking values of total charges column because it is object type
print(df['Total Charges'].unique()) 

#converting object to numeric datatype
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

#again checking datatype and null values in total charges column
print(df['Total Charges'].dtypes)
print(df['Total Charges'].isnull().sum())

#checking for reason of null values in the total charges column
print(df[df['Total Charges'].isnull()][['Tenure Months','Monthly Charges','Total Charges']])

#droping rows with null values 
df=df.dropna(subset=['Total Charges'])
print(df.shape)
print(df['Total Charges'].isnull().sum())


#shape of dataset now is (7032 rows and 32 columns)


#HANDLING CATEGORICAL VALUES (ENCODING)

#Finding object datatype and removing unwanted columns
print(df.select_dtypes(include='object').columns)
df = df.drop(['CustomerID','Country','State','City','Lat Long','Churn Label'], axis=1)
print(df.shape)
print(df.select_dtypes(include='object').columns)





df_clean = df.copy()   # keep cleaned (non-encoded) data for dashboard/risk file
# Save cleaned dataset (before encoding)
df.to_csv("cleaned_telco_data.csv", index=False)
print("Saved: cleaned_telco_data.csv", df.shape)




#shape of dataset now is (7032 rows and 26 columns)

df = pd.get_dummies(df, drop_first=True)
print(df.shape)
print(df.select_dtypes(include='object').columns)
print(df.columns)

#shape of dataset now is (7032 rows and 37 columns)

#removing unwanted columns which creates data leakage and disrupts accuracy of the model

df = df.drop(['Count','Latitude','Longitude','Zip Code','Churn Score'],axis=1)
print('After Encoding' ,df.shape)


#SEPERATING FEATURES AND TARGET VALUES


X = df.drop('Churn Value', axis = 1)
y = df['Churn Value']

print('Feature columns' ,X.shape)
print('Target column' ,y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print('X_train shape', X_train.shape)
print('X_test shape', X_test.shape)
print('y_train shape', y_train.shape)
print('y_test shape', y_test.shape)

print('overall churn rate', y.mean())
print('train churn rate', y_train.mean())
print('test churn rate', y_test.mean())







#MODEL TRAINING PART USING LOGISTIC REGRESSION



print('LOGISTIC REGRESSION')

#training phase

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train) 

#prediction phase

y_pred = model.predict(X_test)

#evaluation phase

from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_test, y_pred))

#classification report

from sklearn.metrics import classification_report
print('classification report', classification_report(y_test, y_pred))

#confusion metrix

from sklearn.metrics import confusion_matrix
print('confusion matrix', confusion_matrix(y_test, y_pred))

#ROC-AUC score

from sklearn.metrics import roc_auc_score
y_prob = model.predict_proba(X_test)[:,1]
print('ROC-AUC:', roc_auc_score(y_test, y_prob))




import numpy as np

# Risk dashboard dataset (business-friendly, non-encoded)
# X_test is a DataFrame, so it has the original row indices
risk_dashboard = df_clean.loc[X_test.index].copy()

risk_dashboard["Churn_Probability"] = y_prob
risk_dashboard["Predicted_Churn"] = (y_prob >= 0.4).astype(int)   # using your optimized threshold
risk_dashboard["Actual_Churn"] = y_test.values

# Optional: Risk segment label
risk_dashboard["Risk_Segment"] = np.where(
    risk_dashboard["Churn_Probability"] >= 0.7, "High",
    np.where(risk_dashboard["Churn_Probability"] >= 0.4, "Medium", "Low")
)

risk_dashboard.to_csv("risk_dashboard.csv", index=False)
print("Saved: risk_dashboard.csv", risk_dashboard.shape)




#MODEL SCALING, RETRAINING AND REEVALUATING



print('FEATURE SCALING')

#scaling phase
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#training phase
#from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 2000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

#evaluating phase
#from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_test, y_pred))

#classification report
#from sklearn.metrics import classification_report
print('Classification report:', classification_report(y_test, y_pred))

#confusion matrix
#from sklearn.metrics import confusion_matrix
print('Confusion matrix:', confusion_matrix(y_test, y_pred))

#ROC-AUC score
#from sklearn.metrics import roc_auc_score
y_prob = model.predict_proba(X_test_scaled)[:,1]
print('ROC-AUC score:', roc_auc_score(y_test, y_prob))



# AGAIN USING RANDOM FOREST MODEL


print('RANDOM FOREST MODEL')

# model training
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# prediction
y_pred_rf = rf_model.predict(X_test)

# accuracy score, classification report, roc-auc score
#from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

print('Accuracy:', accuracy_score(y_test, y_pred_rf))
print('Classification report:', classification_report(y_test, y_pred_rf))

y_prob_rf = rf_model.predict_proba(X_test)[:,1]
print('ROC_AUC score:', roc_auc_score(y_test, y_prob_rf))

print('Confusion matrix:', confusion_matrix(y_test, y_pred_rf))


# Get Probabilities


y_prob_lr = model.predict_proba(X_test_scaled)[:,1]
thresholds = [0.3, 0.4, 0.5, 0.6]

for t in thresholds:
    y_pred_custom = (y_prob_lr >= t).astype(int)
    print(f'\nThreshold: {t}')
    print(classification_report(y_test, y_pred_custom))

    y_prob_final = model.predict_proba(X_test_scaled)[:,1]
    y_pred_final = (y_prob_final >= 0.4).astype(int)


# Cross Validation


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

#pipeline = loogistic regression + scaling (prevents leakage)

lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=3000))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_score = cross_val_score(lr_pipe, X_train, y_train, cv=cv, scoring='roc_auc')
print('CV ROC AUC Score (original)', auc_score)
print('Mean ROC AUC (original)', auc_score.mean())
print('Std ROC AUC (original)', auc_score.std())




lr_balanced = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=3000, class_weight='balanced'))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_score = cross_val_score(lr_balanced, X_train, y_train, cv=cv, scoring='roc_auc')
print('CV ROC AUC Score (balanced)', auc_score)
print('Mean ROC AUC (balanced)', auc_score.mean())
print('Std ROC AUC (balanced)', auc_score.std())




# Class balance check

lr_balanced.fit(X_train, y_train)

y_prob_bal = lr_balanced.predict_proba(X_test)[:,1]
y_pred_bal = (y_prob_bal >= 0.5).astype(int)
print('BALANCED MODEL TEST RESULTS')
print(classification_report(y_test, y_pred_bal))
print('Balanced test roc-auc', roc_auc_score(y_test, y_prob_bal))



# Fit final chosen model (original model not the balanced one)


lr_pipe.fit(X_train, y_train)

coefficients = lr_pipe.named_steps['lr'].coef_[0]
features = X_train.columns

coef_df = pd.DataFrame({
    'Features': features,
    'Coefficient': coefficients
})

coef_df["Absolute_Value"] = np.abs(coef_df["Coefficient"])

coef_df = coef_df.sort_values(by="Absolute_Value", ascending=False)

print("Top 15 Most Important Features:")
print(coef_df.head(15))





import joblib
from datetime import datetime

FINAL_THRESHOLD = 0.4

# Fit pipeline on full training data (important)
lr_pipe.fit(X_train, y_train)

artifact = {
    "model": lr_pipe,
    "threshold": FINAL_THRESHOLD,
    "feature_names": list(X_train.columns),
    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "notes": "Logistic Regression (StandardScaler + LR), threshold optimized to 0.4"
}

joblib.dump(artifact, "churn_model.joblib")
print("Saved model artifact: churn_model.joblib")





# Load saved artifact
artifact = joblib.load("churn_model.joblib")

model = artifact["model"]
threshold = artifact["threshold"]
feature_names = artifact["feature_names"]

print("Model loaded successfully.")
print("Using threshold:", threshold)

# Create a sample input (all zeros for now)
sample_input = pd.DataFrame([[0]*len(feature_names)], columns=feature_names)

# Predict probability
prob = model.predict_proba(sample_input)[0,1]

# Apply threshold
prediction = int(prob >= threshold)

print("Churn Probability:", prob)
print("Prediction:", "CHURN" if prediction == 1 else "NO CHURN")






# Load model artifact
artifact = joblib.load("churn_model.joblib")

model = artifact["model"]
threshold = artifact["threshold"]
feature_names = artifact["feature_names"]

# Use TEST data for risk scoring
X_test_copy = X_test.copy()

# Get probabilities
churn_probabilities = model.predict_proba(X_test_copy)[:, 1]

# Create risk dataframe
risk_df = X_test_copy.copy()
risk_df["Churn_Probability"] = churn_probabilities
risk_df["Predicted_Churn"] = (churn_probabilities >= threshold).astype(int)
risk_df["Actual_Churn"] = y_test.values

# Sort by highest risk
risk_df = risk_df.sort_values(by="Churn_Probability", ascending=False)

print("Top 10 Highest Risk Customers:")
print(risk_df[["Churn_Probability", "Predicted_Churn", "Actual_Churn"]].head(10))

# Export to CSV
risk_df.to_csv("at_risk_customers.csv", index=False)

print("Exported at_risk_customers.csv successfully.")