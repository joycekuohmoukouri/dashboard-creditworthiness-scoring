
# Module créer dans le cadre du projet 7 de la formation OpenClassroom. Ce module rassemble les fonctions qui permettent de préparer le dataset avant l'utilisation du modèle
import pandas as pd
import numpy as np
import scipy.stats as stats
import joblib
import shap 


def application(chemin, selected_features):
  df = pd.read_csv(chemin,
              usecols= selected_features,
              dtype={'SK_ID_CURR' : 'object',
                     #'TARGET' :  'object'
                      })
  # AGE
  df['DAYS_BIRTH'] = round(((df['DAYS_BIRTH']*(-1))/30)/12)
  df['DAYS_BIRTH'] = df['DAYS_BIRTH'].astype(int)
  # ANCIENNETE EMPLOI
  df['DAYS_EMPLOYED'] = round(((df['DAYS_EMPLOYED']*(-1))/30)/12)
  df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].astype(int)
  # RATIO D'ENDETTEMENT
  df['RATIO_ENDETT(%)'] = round((df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'])*100,1)
  # OCCUPATION
  occupation_mapping = {
    'Commercial associate': 'Working',
    'Businessman': 'Working',
    'Maternity leave': 'Working',
    'Student': 'Working'}
  df['NAME_INCOME_TYPE'] = df['NAME_INCOME_TYPE'].map(occupation_mapping)
  # SECTEUR
  organization_categories = {
    'Business Entity Type 3': 'Business Entity',
    'School': 'Education',
    'Government': 'Government and Public Services',
    'Religion': 'Other Categories',
    'Other': 'Other Categories',
    'XNA': 'Inactive',
    'Electricity': 'Utilities and Services',
    'Medicine': 'Healthcare and Medicine',
    'Business Entity Type 2': 'Business Entity',
    'Self-employed': 'Self-Employed and Professional Services',
    'Transport: type 2': 'Other Categories',
    'Construction': 'Other Categories',
    'Housing': 'Utilities and Services',
    'Kindergarten': 'Education',
    'Trade: type 7': 'Industry and Trade',
    'Industry: type 11': 'Industry and Trade',
    'Military': 'Government and Public Services',
    'Services': 'Other Categories',
    'Security Ministries': 'Government and Public Services',
    'Transport: type 4': 'Other Categories',
    'Industry: type 1': 'Industry and Trade',
    'Emergency': 'Government and Public Services',
    'Security': 'Government and Public Services',
    'Trade: type 2': 'Industry and Trade',
    'University': 'Education',
    'Transport: type 3': 'Other Categories',
    'Police': 'Government and Public Services',
    'Business Entity Type 1': 'Business Entity',
    'Postal': 'Government and Public Services',
    'Industry: type 4': 'Industry and Trade',
    'Agriculture': 'Other Categories',
    'Restaurant': 'Hospitality and Entertainment',
    'Culture': 'Hospitality and Entertainment',
    'Hotel': 'Hospitality and Entertainment',
    'Industry: type 7': 'Industry and Trade',
    'Trade: type 3': 'Industry and Trade',
    'Industry: type 3': 'Industry and Trade',
    'Bank': 'Financial and Insurance',
    'Industry: type 9': 'Industry and Trade',
    'Insurance': 'Financial and Insurance',
    'Trade: type 6': 'Industry and Trade',
    'Industry: type 2': 'Industry and Trade',
    'Transport: type 1': 'Other Categories',
    'Industry: type 12': 'Industry and Trade',
    'Mobile': 'Utilities and Services',
    'Trade: type 1': 'Industry and Trade',
    'Industry: type 5': 'Industry and Trade',
    'Industry: type 10': 'Industry and Trade',
    'Legal Services': 'Self-Employed and Professional Services',
    'Advertising': 'Hospitality and Entertainment',
    'Trade: type 5': 'Industry and Trade',
    'Cleaning': 'Utilities and Services',
    'Industry: type 13': 'Industry and Trade',
    'Trade: type 4': 'Industry and Trade',
    'Telecom': 'Utilities and Services',
    'Industry: type 8': 'Industry and Trade',
    'Realtor': 'Self-Employed and Professional Services',
    'Industry: type 6': 'Industry and Trade'}
  df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].replace(organization_categories)
  # Niveau étude
  education_mapping = {
    'Secondary / secondary special': 'BAC',
    'Higher education': 'ENS_SUP',
    'Incomplete higher': 'BAC',
    'Lower secondary': 'COLLEGE',
    'Academic degree': 'ENS_SUP'}
  df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].map(education_mapping)

  #Renommer
  df = df.rename(columns={'DAYS_BIRTH': 'AGE',
                          'DAYS_EMPLOYED' : 'ANCIENNETE_EMPLOI',
                          'ORGANIZATION_TYPE' : 'SECTEUR_ACTIVITE',
                          'NAME_EDUCATION_TYPE' : 'NIVEAU_ETUDE',
                          'NAME_INCOME_TYPE' : 'OCCUPATION',
                          'HOUR_APPR_PROCESS_START' : 'HEURE_APP',
                          'EXT_SOURCE_2' : 'SCORE_2_EXT',
                          'CODE_GENDER' : 'GENRE',
                          'FLAG_OWN_REALTY' : 'PROPRIETAIRE',
                          'CNT_CHILDREN' : 'NBRE_ENFANT',
                          'AMT_INCOME_TOTAL': 'REVENUS_TOT',
                          'AMT_CREDIT' : 'MONTANT_CREDIT',
                          'AMT_ANNUITY' : 'REMB_ANNUEL',
                          'REGION_RATING_CLIENT' : 'SCORE_REGION'
                          })
  return df

def POS_fonct(chemin, df_application):
  df_POS = pd.read_csv(chemin,
                          dtype={'SK_ID_CURR' : 'object',
                                 'SK_ID_PREV' : 'object',})
  # Intersection de la table application et la table credit_card (pour effectuer les traitement uniquement sur les individus concernés)
  df_POS = df_POS.merge(df_application['SK_ID_CURR'], how = 'inner', on = 'SK_ID_CURR')
  df_POS.loc[df_POS['NAME_CONTRACT_STATUS'] != 'Active', 'NAME_CONTRACT_STATUS'] = 'Inactif'

  # Suppression des 0.2% de valeurs manquantes
  df_POS = df_POS[~(df_POS['CNT_INSTALMENT'].isna())]
  #Progession de remboursement
  df_POS['PROGRESS(%)'] = (1- (df_POS['CNT_INSTALMENT_FUTURE']/df_POS['CNT_INSTALMENT']))*100
  progress = df_POS.groupby(['SK_ID_CURR','SK_ID_PREV']).agg({'PROGRESS(%)' : 'max'})
  progress = progress.reset_index()
  progress = progress.groupby('SK_ID_CURR')['PROGRESS(%)'].min()
  progress = progress.reset_index()
  print(progress)
  df_POS = df_POS.drop(columns = ['PROGRESS(%)'])

  #Compte des lignes de crédit---------------------------------------------------------------------------
  count_pos = df_POS.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique()
  count_pos = count_pos.reset_index()
  count_pos = count_pos.rename(columns = {'SK_ID_PREV' : 'N1BRE_CONTRAT'})
  print(count_pos)
  #Compte des lignes de crédits inactives-----------------------------------------------------------------
  pos_inactif = df_POS[df_POS['NAME_CONTRACT_STATUS'] != 'Active']
  count_pos_inactif = pos_inactif.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique()
  count_pos_inactif = count_pos_inactif.reset_index()
  count_pos_inactif = count_pos_inactif.rename(columns = {'SK_ID_PREV' :'N2BRE_CONTRAT_INACTIFS'})
  print(count_pos_inactif)

  # General aggregations----------------------------------------------------------------------------------
  df_POS.drop(['SK_ID_PREV'], axis= 1, inplace = True)
  df_pos_agg = df_POS.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var', 'count']).reset_index()
  del df_POS
  df_pos_agg = df_pos_agg.merge(progress, how = 'left', on ='SK_ID_CURR')
  df_pos_agg = df_pos_agg.merge(count_pos, how = 'left', on ='SK_ID_CURR')
  df_pos_agg = df_pos_agg.merge(count_pos_inactif, how = 'left', on ='SK_ID_CURR')
  df_pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in df_pos_agg.columns.tolist()])
  listcol = df_pos_agg.columns
  # Choix des features -------------------------------------------------------------------------------------
  colum_choice = ['POS_S_K','POS_MONTHS_BALANCE_MIN',
                           'POS_MONTHS_BALANCE_MAX',#'POS_CNT_INSTALMENT_MAX',
                           'POS_SK_DPD_DEF_MEAN','POS_N_1','POS_N_2', 'POS_P_R'
                           ]
  df_pos_agg = df_pos_agg[colum_choice]
  df_pos_agg = df_pos_agg.rename(columns= {'POS_S_K' : 'SK_ID_CURR',
                                         'POS_MONTHS_BALANCE_MIN': 'POS_ANCIENNETE_MOIS',
                                          'POS_MONTHS_BALANCE_MAX': 'POS_RECENCE_MOIS',
                                          #'POS_CNT_INSTALMENT_MAX': 'POS_NBRE_INST_MAX',
                                          'POS_SK_DPD_DEF_MEAN': 'POS_J_RETARD_MOYEN',
                                          'POS_N_1' : 'POS_NBRE_CONTRAT_TOTAL',
                                           'POS_N_2': 'POS_NBRE_CONTRAT_INACTIF',
                                           'POS_P_R' : 'POS_PROGRESS_MAX_MIN'})

  df_pos_agg[['POS_ANCIENNETE_MOIS', 'POS_RECENCE_MOIS']] = df_pos_agg[['POS_ANCIENNETE_MOIS', 'POS_RECENCE_MOIS']]*(-1)
  df_pos_agg['POS_NBRE_CONTRAT_INACTIF'] = df_pos_agg['POS_NBRE_CONTRAT_INACTIF'].fillna(0)
  df_pos_agg['POS_NBRE_CONTRAT_INACTIF'] = df_pos_agg['POS_NBRE_CONTRAT_INACTIF'].astype(int)


  return df_pos_agg



# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(chemin_,df, num_rows = None):
    bureau = pd.read_csv(chemin_+'/bureau.csv', #nrows = 10,
                         dtype={'SK_ID_CURR' : 'object',
                                'SK_BUREAU_ID' : 'object',
                             })
    #print('check 0')
    bureau = bureau.merge(df['SK_ID_CURR'], how = 'inner', on = 'SK_ID_CURR')
    #print('check merge')
    ####------------------------- Bureau balance
    bb = pd.read_csv(chemin_+'/bureau_balance.csv', #nrows = 10,
                     dtype={'SK_BUREAU_ID' : 'object',
                             })
    #print('check 1')
    bb_aggregations = {'MONTHS_BALANCE': ['max', 'min']}
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations).reset_index()
    del bb
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    print(bb_agg.columns)
    print(bb_agg)
    bureau = bureau.merge(bb_agg, how='left', left_on='SK_ID_BUREAU', right_on = 'SK_ID_BUREAU_')
    #bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb_agg

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['max'],
        #'DAYS_CREDIT_ENDDATE': ['max'],
        'CREDIT_DAY_OVERDUE': ['max'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['sum'],
        'AMT_ANNUITY': ['sum'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_MIN': ['min'],
        'SK_ID_BUREAU_' : ['count']
    }
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(num_aggregations).reset_index()
    bureau_agg.columns = pd.Index(['BUR_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    print(bureau_agg.columns)
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations).reset_index()
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.merge(active_agg, how='left', left_on = 'BUR_SK_ID_CURR_', right_on='ACTIVE_SK_ID_CURR_')
    del active, active_agg
    print(bureau_agg.columns)
    bureau_agg = bureau_agg[['BUR_SK_ID_CURR_','ACTIVE_DAYS_CREDIT_MAX',
                             #'ACTIVE_DAYS_CREDIT_ENDDATE_MAX',
                             'BUR_CREDIT_DAY_OVERDUE_MAX', 'BUR_AMT_CREDIT_SUM_OVERDUE_SUM',
                             'BUR_MONTHS_BALANCE_MIN_MIN','ACTIVE_SK_ID_BUREAU__COUNT', 'BUR_SK_ID_BUREAU__COUNT',
                             'ACTIVE_AMT_ANNUITY_SUM', 'ACTIVE_MONTHS_BALANCE_MAX_MAX','ACTIVE_AMT_CREDIT_SUM_DEBT_SUM']]

    bureau_agg = bureau_agg.rename(columns = {'BUR_SK_ID_CURR_' : 'SK_ID_CURR',
                                              'ACTIVE_DAYS_CREDIT_MAX': 'CB_RECENCE_APPL',
                                             #'ACTIVE_DAYS_CREDIT_ENDDATE_MIN' : 'CB_J_RESTANT',
                                              'BUR_CREDIT_DAY_OVERDUE_MAX' : 'CB_J_RETARD',
                                              'BUR_MONTHS_BALANCE_MIN_MIN' : 'CB_ANCIENNETE_MOIS',
                                              'ACTIVE_SK_ID_BUREAU__COUNT' : 'CB_NBRE_CONTRAT_ACTIF',
                                              'BUR_SK_ID_BUREAU__COUNT': 'CB_NBRE_CONTRAT_TOTAL',
                                              'ACTIVE_AMT_ANNUITY_SUM' : 'CB_REMB_ANNUEL_TOTAL',
                                              'ACTIVE_MONTHS_BALANCE_MAX_MAX' : 'CB_RECENCE_ACTIVITE',
                                              'ACTIVE_AMT_CREDIT_SUM_DEBT_SUM' : 'CB_RESTE_A_PAYER',
                                              'BUR_AMT_CREDIT_SUM_OVERDUE_SUM' : 'CB_SOMME_DUES_RETARD'
                                             })
    return bureau_agg

def credit_card_fonct(chemin, selected_features,df_application):
  df_credit_card = pd.read_csv(chemin,
                          usecols= selected_features,
                          dtype={'SK_ID_CURR' : 'object',
                                 'SK_ID_PREV' : 'object',})
  # Intersection de la table application et la table credit_card (pour effectuer les traitement uniquement sur les individus concernés)
  df_credit_card = df_credit_card.merge(df_application['SK_ID_CURR'], how = 'inner', on = 'SK_ID_CURR')
  df_credit_card.loc[df_credit_card['NAME_CONTRACT_STATUS'] != 'Active', 'NAME_CONTRACT_STATUS'] = 'Inactif'

  #
  df_credit_card['AMT_INST_MIN_REGULARITY'] = df_credit_card['AMT_INST_MIN_REGULARITY'].fillna(0)

  #Compte des lignes de crédit---------------------------------------------------------------------------
  count_cc = df_credit_card.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique()
  count_cc = count_cc.reset_index()
  count_cc = count_cc.rename(columns = {'SK_ID_PREV' : 'N1BRE_CONTRAT'})
  print(count_cc)
  #Compte des lignes de crédits inactives-----------------------------------------------------------------
  cc_inactif = df_credit_card[df_credit_card['NAME_CONTRACT_STATUS'] != 'Active']
  count_cc_inactif = cc_inactif.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique()
  count_cc_inactif = count_cc_inactif.reset_index()
  count_cc_inactif = count_cc_inactif.rename(columns = {'SK_ID_PREV' :'N2BRE_CONTRAT_INACTIFS'})
  print(count_cc_inactif)
  # General aggregations----------------------------------------------------------------------------------
  df_credit_card.drop(['SK_ID_PREV'], axis= 1, inplace = True)
  df_cc_agg = df_credit_card.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var', 'count']).reset_index()
  del df_credit_card
  df_cc_agg = df_cc_agg.merge(count_cc, how = 'left', on ='SK_ID_CURR')
  df_cc_agg = df_cc_agg.merge(count_cc_inactif, how = 'left', on ='SK_ID_CURR')
  df_cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in df_cc_agg.columns.tolist()])
  listcol = df_cc_agg.columns
  # Choix des features -------------------------------------------------------------------------------------
  df_cc_agg = df_cc_agg[['CC_S_K', 'CC_MONTHS_BALANCE_MIN', 'CC_MONTHS_BALANCE_MAX',
                 'CC_AMT_BALANCE_MEAN', 'CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN',
                 'CC_AMT_DRAWINGS_CURRENT_MEAN', 'CC_AMT_INST_MIN_REGULARITY_MEAN',
                 'CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN', 'CC_CNT_DRAWINGS_CURRENT_MEAN',
                 'CC_SK_DPD_DEF_MEAN', 'CC_N_1', 'CC_N_2'
                 ]]
  df_cc_agg = df_cc_agg.rename(columns= { 'CC_S_K' : 'SK_ID_CURR',
                                         'CC_MONTHS_BALANCE_MIN': 'CC_ANCIENNETE_MOIS',
                                          'CC_MONTHS_BALANCE_MAX': 'CC_RECENCE_MOIS',
                                          'CC_AMT_BALANCE_MEAN': 'CC_SOLDE_MOYEN_MENSUEL',
                                          'CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN': 'CC_LIMITE_CREDIT_MOYENNE',
                                          'CC_AMT_DRAWINGS_CURRENT_MEAN': 'CC_MONTANT_RETRAIT_MOYEN',
                                          'CC_AMT_INST_MIN_REGULARITY_MEAN': 'CC_MIN_REMBOURSEMENT_MOYEN',
                                          'CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN': 'CC_PAIEMENT_TOTAL_MOYEN',
                                          'CC_CNT_DRAWINGS_CURRENT_MEAN': 'CC_NOMBRE_RETRAIT_MOYEN',
                                          'CC_SK_DPD_DEF_MEAN': 'CC_J_RETARD_MOYEN',
                                          'CC_N_1' : 'CC_NBRE_CONTRAT_TOTAL', 'CC_N_2': 'CC_NBRE_CONTRAT_INACTIF'})
  df_cc_agg[['CC_ANCIENNETE_MOIS', 'CC_RECENCE_MOIS']] = df_cc_agg[['CC_ANCIENNETE_MOIS', 'CC_RECENCE_MOIS']]*(-1)
  df_cc_agg['CC_NBRE_CONTRAT_INACTIF'] = df_cc_agg['CC_NBRE_CONTRAT_INACTIF'].fillna(0)
  df_cc_agg['CC_NBRE_CONTRAT_INACTIF'] = df_cc_agg['CC_NBRE_CONTRAT_INACTIF'].astype(int)
  return df_cc_agg, listcol

def install_fonct(chemin, df):
  def filtre(value):
    if value < 0:
      return 0
    else :
      return value
  df_install = pd.read_csv(chemin,
                           #nrows = 10,
                      dtype={'SK_ID_CURR' : 'object',
                             'SK_ID_PREV' : 'object',
                             #'DAYS_ENTRY_PAYMENT' :'float'
                             }
                      )
  df_install = df_install.merge(df[['SK_ID_CURR']], how = 'inner', on = 'SK_ID_CURR')
  df_install['DAYS_ENTRY_PAYMENT'] = df_install['DAYS_ENTRY_PAYMENT'].astype(float)
  #Je supprime les paiement de montant nul, ils font partis d'une série de paiement.
  #Ce qui me confirme que les valeur manquantes sont bien des paiements manqués.
  #install[(install['AMT_PAYMENT'] == 0)]
  ind_ = df_install[(df_install['AMT_PAYMENT'] == 0)].index
  df_install = df_install.drop(ind_)
  df_install = df_install.fillna(0)
  df_install['J_RETARD'] = df_install['DAYS_ENTRY_PAYMENT'] - df_install['DAYS_INSTALMENT']
  df_install['J_RETARD'] = df_install['J_RETARD'].map(filtre)
  df_install = df_install[['SK_ID_PREV',
                           'SK_ID_CURR','J_RETARD', 'NUM_INSTALMENT_NUMBER',
                           'AMT_INSTALMENT', 'AMT_PAYMENT']]
  install_agg = df_install.groupby(['SK_ID_CURR','SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']).agg({'J_RETARD': 'max',
                                                                                              'AMT_INSTALMENT': ['max'],
                                                                                              'AMT_PAYMENT': ['sum'], }).reset_index()
  del df_install
  install_agg.columns = pd.Index(['INST_' + e[0] + "_" + e[1].upper() for e in install_agg.columns.tolist()] )

  install_agg['INST_RESTE_A_PAYER'] = install_agg['INST_AMT_INSTALMENT_MAX']- install_agg['INST_AMT_PAYMENT_SUM']
  install_agg = install_agg.drop(columns =['INST_SK_ID_PREV_', 'INST_NUM_INSTALMENT_NUMBER_'])
  install_agg = install_agg.groupby('INST_SK_ID_CURR_').agg({'INST_J_RETARD_MAX': 'mean',
                                                             'INST_RESTE_A_PAYER' : 'sum'}).reset_index()
  install_agg = install_agg.rename(columns = {'INST_SK_ID_CURR_': 'SK_ID_CURR',
                                             })
  print(install_agg.columns)
  print(install_agg)
  return install_agg

def merge_(df, credit_card, POS, install, bureau):
  df = df.merge(credit_card, how = 'left', on = 'SK_ID_CURR')
  df = df.merge(POS, how = 'left', on = 'SK_ID_CURR')
  df = df.merge(install, how = 'left', on = 'SK_ID_CURR')
  df = df.merge(bureau, how = 'left', on = 'SK_ID_CURR')
  ## RATIO entre le montant du crédit demandé et l'income annuel 
  df['RATIO_CREDIT_REVENU'] = round((df['MONTANT_CREDIT']/df['REVENUS_TOT']),2)
  ## RATIO CARTE DE CREDIT
  df['CC_RATIO_CREDIT'] = round((df['CC_SOLDE_MOYEN_MENSUEL']/df['CC_LIMITE_CREDIT_MOYENNE'])*100,2)
  df['CC_RATIO_CREDIT'] = df['CC_RATIO_CREDIT'].fillna(0)
  ## Pour les clients n'ayant pas de carte de crédit, les features suivantes sont nulles
  col_cc = ['CC_ANCIENNETE_MOIS', 'CC_RECENCE_MOIS',
       'CC_SOLDE_MOYEN_MENSUEL', 'CC_LIMITE_CREDIT_MOYENNE',
       'CC_MONTANT_RETRAIT_MOYEN', 'CC_MIN_REMBOURSEMENT_MOYEN',
       'CC_PAIEMENT_TOTAL_MOYEN', 'CC_NOMBRE_RETRAIT_MOYEN',
       'CC_J_RETARD_MOYEN', 'CC_NBRE_CONTRAT_TOTAL',
       'CC_NBRE_CONTRAT_INACTIF']
  df[col_cc] = df[col_cc].fillna(0)
  #idem POS
  col_pos = ['POS_ANCIENNETE_MOIS', 'POS_RECENCE_MOIS',
       'POS_J_RETARD_MOYEN', 'POS_NBRE_CONTRAT_TOTAL',
       'POS_NBRE_CONTRAT_INACTIF', 'POS_PROGRESS_MAX_MIN']
  df[col_pos] = df[col_pos].fillna(0)

  #idem credit_bureau
  col_bur = ['CB_RECENCE_APPL', 'CB_J_RETARD', 'CB_SOMME_DUES_RETARD',
       'CB_ANCIENNETE_MOIS', 'CB_NBRE_CONTRAT_ACTIF', 'CB_NBRE_CONTRAT_TOTAL',
       'CB_REMB_ANNUEL_TOTAL', 'CB_RECENCE_ACTIVITE', 'CB_RESTE_A_PAYER']
  df[col_bur] = df[col_bur].fillna(0)

  #idem install
  col_inst= ['INST_J_RETARD_MAX', 'INST_RESTE_A_PAYER']
  df[col_inst] = df[col_inst].fillna(0)
  return df

def feat_engineering(df):
  df['NBRE_CONTRAT_ACTIFS'] = (df['CC_NBRE_CONTRAT_TOTAL']-df['CC_NBRE_CONTRAT_INACTIF']) + (df['POS_NBRE_CONTRAT_TOTAL']-df['POS_NBRE_CONTRAT_INACTIF']) + (df['CB_NBRE_CONTRAT_ACTIF'])
  df['NBRE_J_RETARD'] = df['CB_J_RETARD'] + df['INST_J_RETARD_MAX'] + df['CC_J_RETARD_MOYEN'] + df['POS_J_RETARD_MOYEN']
  df['CHARGES_ANNUEL'] = (df['CC_MIN_REMBOURSEMENT_MOYEN']*12 + df['CB_REMB_ANNUEL_TOTAL'] + df['INST_RESTE_A_PAYER'] + df['REMB_ANNUEL'])
  df['RATIO_ENDETT_1(%)'] = (df['CHARGES_ANNUEL']/ (df['REVENUS_TOT']))*100
  df['ANCIENNETE_CREDIT'] = np.max(np.abs(df[['CC_ANCIENNETE_MOIS', 'POS_ANCIENNETE_MOIS', 'CB_ANCIENNETE_MOIS']]), axis=1)
  return df

def nettoyage(df):
  def supp_outliers(df, FEAT, lim_fact, val):
      iqr = stats.iqr(df[FEAT])
      lim = iqr*lim_fact
      df.loc[df[FEAT] >= iqr+ lim, FEAT] = val
      return df
  df = supp_outliers(df, 'NBRE_ENFANT', 4, 0)
  df = supp_outliers(df, 'REVENUS_TOT',10, np.nan)
  df = supp_outliers(df, 'CHARGES_ANNUEL',10, np.nan)
  df.loc[df['CC_RATIO_CREDIT'] == np.inf, 'CC_RATIO_CREDIT']= np.nan
  df = df[~(df['REVENUS_TOT'].isna())]
  df = df[~(df['REMB_ANNUEL'].isna())]
  df['ANCIENNETE_EMPLOI'].replace(-1015, 0, inplace=True)
  df['GENRE'].replace('XNA', 'F', inplace = True)
  return df

###----------------------------- Fonctions pour l'API---------------------------------------


# Fonction permettant de charger le preprocessing
def preprocess_model():
    from projet7package.frequency_encode import frequency_encode
    from sklearn.preprocessing import FunctionTransformer
    freq_encoder = FunctionTransformer(frequency_encode)
    with open('preprocessing_2.pkl', 'rb') as f:
        loaded_preprocess = joblib.load(f)
    return loaded_preprocess

# Fonction pour obtenir les données clients
def get_client_data(client_id):
    selected_features = [#'GENRE',
                     'PROPRIETAIRE', 'NBRE_ENFANT',
                     'ANCIENNETE_CREDIT', 'CHARGES_ANNUEL', 'REVENUS_TOT',
                     #'MONTANT_CREDIT',
                    'RATIO_CREDIT_REVENU',
                     'OCCUPATION', 'CC_RATIO_CREDIT', 'NIVEAU_ETUDE', 'AGE',
                     'ANCIENNETE_EMPLOI', 'SCORE_REGION', 'HEURE_APP', 'SECTEUR_ACTIVITE',
                     'SCORE_2_EXT',
                     #'RATIO_ENDETT(%)',
                     'NBRE_CONTRAT_ACTIFS','NBRE_J_RETARD', 'POS_PROGRESS_MAX_MIN',
                     'CC_NOMBRE_RETRAIT_MOYEN', 'CB_SOMME_DUES_RETARD']
    df = pd.read_csv('client_test_db.csv',dtype={'SK_ID_CURR' : 'object'})
    client_data = df[df['SK_ID_CURR'] == client_id]
    client_data = client_data[selected_features]
    return client_data

def feat_local(df_client_pp):
        classification_model = joblib.load('LightGBM_bestmodel.pkl')
        # J'instancie le Shap explainer -----------------------------------
        explainer = shap.TreeExplainer(classification_model)
        # Calculate SHAP values for the client's prediction
        shap_values = explainer.shap_values(df_client_pp)
        ##-------- df_pp
        prefixes_to_remove = ['oneHot__', 'remainder__', 'frequency__']
        new_column_names = [col.replace(prefix, '') for col in df_client_pp.columns for prefix in prefixes_to_remove if col.startswith(prefix)]
        df_client_pp.columns = new_column_names
        return shap_values, df_client_pp


