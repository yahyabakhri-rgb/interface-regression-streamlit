### Importer les librairies python nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # pour enregistrer et télécharger le modèle/ ou scaler 
from sklearn.pipeline import Pipeline #pour automatiser les étapes de prétraitement et de modélisation interessantes pour les nouvelles données

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet, RidgeCV,LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
#
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#lire la base de données
data=pd.read_csv('prostate_dataset.txt',delimiter="\t")

print(f" data.shape:{data.shape}")
# Afficher l’en-tête des données
data.head()

# Afficher la liste des  variables
data.columns
# Afficher le type de  chaque variable
data.dtypes

col_sup=["col",'train'] # colonnes à supprimer 
data1=data.drop(col_sup,axis=1)   
data1.head()
# Vérifier s'il y a des données manquantes
data1.isna().mean()

# Vérifier s'il y a des données manquantes:visualisation graphique
plt.figure(figsize=(4,3))
sns.heatmap(data1.isna())

## Décrire les variables qualitatives séparément :
qualitative_vars=['svi','gleason']
for col in qualitative_vars:
    print(f"Distribution des catégories pour {col}:")
    print(data1[col].value_counts())
    print("\n")

## séparer la variables à expliquer et les variables explicatives
y=data1["lpsa"] # variable target  
X=data1.drop(["lpsa"],axis=1)

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
def eval_regression(y,y_pred,label=""):
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae  = mean_absolute_error(y, y_pred)
    r2   = r2_score(y, y_pred)
    dic={"model": label, "RMSE": rmse, "MAE": mae, "R2": r2}
    return  dic

model=LinearRegression()
model.fit(x_train,y_train)
print(x_train.columns)
print(f"model.coef_:{model.coef_}")

#pour automatiser les étapes de prétraitement et de modélisation interessantes pour les nouvelles données
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ('reg',LinearRegression())
])
pipe.fit(x_train,y_train) #entraine la pipeline

# Sauvegarder le modèle entraîné
joblib.dump(pipe, "Model_MIS.pkl")

print("Modèle sauvegardé avec succès !")

pipe.named_steps['reg'].intercept_
pipe.named_steps['reg'].coef_

def print_coef(x_train,pipe):
    coef_df = pd.DataFrame({
    "Variable": x_train.columns,
    "Coefficient": pipe.named_steps['reg'].coef_
    })
    coef_df =coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index)
    print(coef_df)
# La méthode reindex() permet de réorganiser les lignes du DataFrame
# selon les indices fournis.
# .index permet de récupérer les nouveaux indices après le tri

# Représentation des résidus
def plot_residuals(y, pred, title):
    resid = y- pred
    plt.figure(figsize=(3,3))
    plt.scatter(pred, resid, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Prédictions")
    plt.ylabel("Résidus (y - ŷ)")
    plt.title(title)
    plt.show()

print_coef(x_train,pipe)
# tester la performance du  modèle
pred1=pipe.predict(x_test)
dic1=eval_regression(y_test,pred1,"lineare regression")
dic1

#Visualizing Predictions
plt.figure(figsize=(3,3))
plt.scatter(y_test, pred1)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs Predicted Values")
plt.show()

plot_residuals(y_test, pred1, "OLS")

# Les nouvelles données de prédiction
new={
    'lcavol':[2.8, 3],
    'lweight':[3,2],
    'age':[70,50],
    'lbph':[-1.4,-1],
    'svi':[1,0],
    'lcp':[1.5,1.7],
    'gleason':[7,9],
    'pgg45':[60,70],
    
}

df=pd.DataFrame(new)