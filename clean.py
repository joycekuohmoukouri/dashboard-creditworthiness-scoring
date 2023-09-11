# Module qui rassemble des fonctions qui me permettent de nettoyer un dataframe
#---------------------------------------------------------------------------------

import missingno as msno
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


#--------------------------------------------------------------------------------
#Data sampling
#def randselec(df,n
#--------------------number of records in file e.g n = 1000000
#--------------------s = 10000 #desired sample size
#filename = "data.txt"
#skip = sorted(random.sample(range(n),n-s))
#df = pandas.read_csv(filename, skiprows=skip)

#--------------------------------------------------------------------------------
#Visualisation des infos
def info(df):
    #Renvoie les info realtives à df
    print(df.info()) #taille, type, mémoire 
    print(df.describe())
    df.head()
    return 

#---------------------------------------------------------------------------------
#Traitement des valeurs abberantes
def val_aberrante(df,col,val_min, val_max):
    if (df[col].max() > val_max):
        ind = df[df[col] > val_max].index
        for i in ind:
            df.loc[i,col] = np.nan   
    else:
        print(col,'Pas de valeurs >', val_max)
    if (df[col].min() < val_min):
        ind = df[df[col] < val_min].index
        for i in ind:
            df.loc[i,col] = np.nan
    else:
        print(col,'Pas de valeurs <', val_min)
    
    return df
#---------------------------------------------------------------------------------
#Visualisation des valeurs manquantes 

def vm(df,t,nomfich):
    #Renvoie les informations relatives au valeurs manquantes du Dataframe df 
    #t, taux du sampling entre 0 et 1
    #nomfich, le nom de fichier attribué au plot 
    l = len(df)*t
    l = round(l)
    A = df.sample(l) 
    #%matplotlib inline
    msno.matrix(A)
    tick = np.arange(df.shape[1])
    lab = df.columns.tolist()
    #plt.title('Visualisation des valeurs manquantes de la table Data',pad = 0, fontsize = 20)
    plt.xticks(tick,labels = lab, rotation='vertical',fontsize = 13 )
    vm_Df = df.isna().mean() 
    plt.savefig(nomfich,dpi = 200, bbox_inches = 'tight')
    return vm_Df

#---------------------------------------------------------------------------------
#Homogénéisation de mon dataframe : tous les str. en upper case + supression des espaces et caractères spéciaux 
def supp_esp(df,col): #série composé de valeur str
    A_space = df[df[col].str.startswith(' ').fillna(False)]
    #print(A_space)
    if A_space.shape[0] == 0:
        print('aucun espace indésirable (amont)')
    else: 
        print(A_space.shape[0],'individus identifiés avec espace indésirable (amont)')
        for i, values in A_space[col].items():
            while values.startswith(' '):
                values = values[1:]
            df[col].loc[i] = values
#-------------------------------        
    A_space = df[df[col].str.endswith(' ').fillna(False)]
    #print(A_space)
    if A_space.shape[0] == 0:
        print('aucun espace indésirable (aval)')
    else: 
        print(A_space.shape[0],'individus identifiés avec espace indésirable (aval)')
        for i, values in A_space[col].items():
            while values.endswith(' '):
                values = values[:-1]
            df[col].loc[i] = values
    return df[col]
    
def homostr(df):
    df.columns = df.columns.str.upper()
    # 1- Sélection des variables de type object (ou str) et passage en lettres capitales
    var_obj = df.dtypes[df.dtypes == object].index    
    for i in var_obj:    
        print('variable',i)
        df[i] = [str(x) for x in df[i]]
        df[i] = df[i].str.upper()
        df[i] = df[i].str.strip()
    return df

#---------------------------------------------------------------------------------

def vide(df,ls_col):
    #____ Remplace les cases vides type '' par np.nan
    for i in ls_col:
        A = df[df[i] == '']
        if A.shape[0] > 0:
            print(A.shape[0],'case(s) vide(s), variable', i, 'traitée')
            df.loc[df[i] == '',i] =np.nan       
        else: 
            print('Aucune case vide, variable', i, 'traitée')
    return df

#---------------------------------------------------------------------------------
#Récupération des colonnes 

def seleccol(df,taux):
    #_____ Retourne la liste des colonnes de df qui ont un taux de valeurs manquantes str. inférieur à taux
    df_vm = df.isna().mean()
    df_vm = df_vm[df_vm <taux]
    print('Il y a',df_vm.shape[0],'colonnes qui ont un taux de valeurs manquantes inférieur à',taux*100,'%')
    col = df_vm.index.values.tolist()
    print('Liste des colonnes sélectionnées',col)
    return col
#---------------------------------------------------------------------------------
#Traitement des doublons
def doublon(df,ls_col):
    #Recherche les doublons, ls_col liste des colonnes qui constitue une clé primaire
    print('Recherche de doublon : il y a ', 
    df.duplicated(ls_col,keep=False).sum(),
      '\ndoublons qui ont la même clé:',ls_col )
#------------------------------  
    #Suppression des doublons
    # on compte le nombre de valeurs manquantes pour la ligne et on stocke dans une nouvelle colonne
    df['NB_NAN'] = df.isna().sum(axis=1)
    # trie des lignes en fonction du nombre de valeurs manquantes
    df= df.sort_values('NB_NAN')
    # suppression des duplicatas en gardant les versions les mieux remplies
    df= df.drop_duplicates(ls_col, keep='first')
    # on supprime la colonne qui n'est plus utile
    df= df.drop('NB_NAN', axis=1)
    df.head()   
    return df

#---------------------------------------------------------------------------------
#Traitement des valeurs manquantes 