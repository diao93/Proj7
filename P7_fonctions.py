import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.decomposition import PCA


def data_duplicated(df):
    '''Retourne le nombres de lignes identiques.'''
    return df.duplicated(keep=False).sum()

def row_duplicated(df,col):
    '''Retourne le nombre de doublons de la variables col.'''
    return df.duplicated(subset = col, keep='first').sum()
    
def missing_cells(df):
    '''Calcule le nombre de cellules manquantes sur le data set total'''
    return df.isna().sum().sum()

def missing_cells_perc(df):
    '''Calcule le pourcentage de cellules manquantes sur le data set total'''
    return df.isna().sum().sum()/(df.size)
    
def missing_general(df):
    '''Donne un aperçu général du nombre de données manquantes dans le data frame'''
    print('Nombre total de cellules manquantes :',missing_cells(df))
    print('Nombre de cellules manquantes en % : {:.2%}'.format(missing_cells_perc(df)))
    
def valeurs_manquantes(df):
    '''Prend un data frame en entrée et créer en sortie un dataframe contenant le nombre de valeurs manquantes
    et leur pourcentage pour chaque variables. '''
    tab_missing = pd.DataFrame(columns = ['Variable', 'Missing values', 'Missing (%)'])
    tab_missing['Variable'] = df.columns
    missing_val = list()
    missing_perc = list()
    
    for var in df.columns:
        nb_miss = missing_cells(df[var])
        missing_val.append(nb_miss)
        perc_miss = missing_cells_perc(df[var])
        missing_perc.append(perc_miss)
        
    tab_missing['Missing values'] = list(missing_val)
    tab_missing['Missing (%)'] = list(missing_perc)
    return tab_missing

def bar_missing(df):
    '''Affiche le barplot présentant le nombre de données présentes par variable.'''
    msno.bar(df)
    plt.title('Nombre de données présentes par variable', size=15)
    plt.show()
    
def barplot_missing(df):
    '''Affiche le barplot présentant le pourcentage de données manquantes par variable.'''
    proportion_nan = df.isna().sum().divide(df.shape[0]/100).sort_values(ascending=False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 30))
    ax = sns.barplot(y = proportion_nan.index, x=proportion_nan.values)
    plt.title('Pourcentage de données manquantes par variable', size=15)
    plt.show()
    
def drop_columns_empty(df,lim):
    '''Prend en entrée un data frame et un seuil de remplissage de données.
    Supprime chaque variable ayant un pourcentage de données manquantes supérieur à celui renseigné. 
    Donne en sortie le data frame filtré avec les colonnes à garder.'''
    
    tab = valeurs_manquantes(df)
    columns_keep = list()
    for row in tab.iterrows():
        if float(row[1]['Missing (%)'])>float(lim):
            print('Suppression de la variable {} avec % de valeurs manquantes {}'.format(row[1]['Variable'],
                                                                                         round(float(row[1]['Missing (%)']),2)))
            
        else :
            columns_keep.append(row[1]['Variable'])
    
    return df[columns_keep]   

def drop_columns_empty(df,lim):
    '''Prend en entrée un data frame et un seuil de remplissage de données.
    Supprime chaque variable ayant un pourcentage de données manquantes supérieur à celui renseigné. 
    Donne en sortie le data frame filtré avec les colonnes à garder.'''
    
    tab = valeurs_manquantes(df)
    columns_keep = list()
    for row in tab.iterrows():
        if float(row[1]['Missing (%)'])>float(lim):
            print('Suppression de la variable {} avec % de valeurs manquantes {}'.format(row[1]['Variable'],
                                                                                         round(float(row[1]['Missing (%)']),2)))
            
        else :
            columns_keep.append(row[1]['Variable'])
    
    return df[columns_keep]

def boxplot(df,ylim):
    ''' Affiche une fenêtre contenant tous les boxplots des variables sélectionnées'''
    fig = plt.figure(figsize=(20, 15))
    ax = plt.axes()
    plt.xticks(rotation=90)
    ax.set_ylim(ylim)
    sns.boxplot(data=df)
    plt.title('Boxplot des variables', size=15)
    
def multi_boxplot(df):
    ''' Affiche indépendamment tous les boxplots des variables sélectionnées'''
    fig, axs = plt.subplots(4,3,figsize=(20,20))
    axs = axs.ravel()
    
    for i,col in enumerate(df.columns):
        sns.boxplot(x=df[col], ax=axs[i])
    fig.suptitle('Boxplot pour chaque variable quantitative')
    plt.show()
        
def distribution(df,colonnes,n_cols,nom,fig=(20,20)):
    ''' Affiche les histogrammes pour chaque variable renseignée.'''
    n_rows = int(len(colonnes)/n_cols)+1
    fig = plt.figure(figsize=fig)
    for i, col in enumerate(colonnes,1):
        ax = fig.add_subplot(n_rows,n_cols,i)
        sns.histplot(data=df, x=col, bins=30, kde=True, ax=ax)

    plt.tight_layout(pad = 2)
    plt.savefig(nom)
    plt.show()
    
def bar_plot(df,colonnes,n_cols,nom,fig=(20,20)):
    ''' Affiche les bar plots pour chaque variable renseignée.'''
    fig = plt.figure(figsize=fig)
    n_rows = int(np.ceil(len(colonnes)/n_cols))
    for i, col in enumerate(colonnes,1):
        ax = fig.add_subplot(n_rows,n_cols,i)
        count = df[col].value_counts()
        count.plot(kind="bar", ax=ax, fontsize=20, rot=90)
        ax.set_title(col, fontsize = 20)
    plt.tight_layout(pad = 2)
    plt.savefig(nom)
    plt.show()

def bar_plot_stacked(df,colonnes,n_cols,nom,fig=(20,20)):
    ''' Affiche les bar plots pour chaque variable renseignée décomposés en fonction de var2.'''
    fig = plt.figure(figsize=fig)
    n_rows = int(np.ceil(len(colonnes)/n_cols))
    for i, col in enumerate(colonnes,1):
        ax = fig.add_subplot(n_rows,n_cols,i)
        count = pd.DataFrame(df.groupby(col)['TARGET'].value_counts()).reset_index()
        count = count.pivot_table(index=col, columns = 'TARGET', values = 'count')
        #count = pd.crosstab(index=count[col], columns =count[var2], values = 'count')
        count.plot(kind="bar", stacked=True, ax=ax, fontsize=20, rot=90)
        ax.set_title(col, fontsize = 20)
        ax.legend(['rembourse','défaut'],fontsize = 20)
    plt.tight_layout(pad = 2)
    plt.savefig(nom)
    plt.show()

def pie_plot(df,colonnes):
    '''Affiche un pie plot présentant la répartition de la variable renseignée.'''
    for col in colonnes :
        labels = list(df[col].value_counts().sort_index().index.astype(str))
        count = df[col].value_counts().sort_index()
        
        plt.figure(figsize=(5, 5))
        plt.pie(count,autopct='%1.2f%%')
        plt.title('Répartition de {}'.format(col), size = 20)
        plt.legend(labels)
        plt.show()

def distribution_densite(df,colonnes,n_cols,nom,fig=(20,20)):
    ''' Affiche les densités  pour chaque variable renseignée.'''
    n_rows = int(len(colonnes)/n_cols)+1
    fig = plt.figure(figsize=fig)
    for i, col in enumerate(colonnes,1):
        ax = fig.add_subplot(n_rows,n_cols,i)
        sns.kdeplot(df.loc[df['TARGET'] == 0, col], label = 'rembourse')
        sns.kdeplot(df.loc[df['TARGET'] == 1, col], label = 'défaut')
        ax.set_xlabel(col, fontsize=20)
        ax.set_ylabel('Density', fontsize=20)
        ax.legend(fontsize=20)
        ax.set_title('Distribution of '+col, fontsize=20)
        
    plt.tight_layout(pad = 2)
    plt.savefig(nom)
    plt.show()

def scatter_plot(df,colonnes,var_comparaison, largeur, longueur):
    ''' Affiche le scatter plot des variables quantitatives.'''
    fig = plt.figure(figsize=(15,15))
    for i,col in enumerate(colonnes,1):
        X = df[[var_comparaison]]
        Y = df[col]
        X = X.copy()
        X['intercept'] = 1.
        result = sm.OLS(Y, X).fit()
        a,b = result.params[var_comparaison],result.params['intercept']
        equa = "y = " + str(round(a,2)) + " x + " + str(round(b,0))

        ax = fig.add_subplot(longueur,largeur,i)
        plt.scatter(x=df[var_comparaison], y=df[col])        
        plt.plot(range(-15,41),[a*x+b for x in range(-15,41)],label=equa,color='red')
        ax.set_xlabel(xlabel=var_comparaison)
        ax.set_ylabel(ylabel=col)
        plt.legend()
    plt.tight_layout(pad = 4)
    fig.suptitle("Scatter plot des variables quantitatives")
    plt.show()
    
def heat_map(df_corr):
    '''Affiche la heatmap '''
    plt.figure(figsize=(30,30))
    sns.heatmap(df_corr, annot=True, linewidth=.5)
    plt.title("Heatmap")

def boxplot_relation(df,colonnes,var_comparaison,longueur,largeur, ordre=None,outliers=True,option=False):
    '''Affiche les boxplot des colonnes en fonctions de var_comparaison.'''
    fig = plt.figure(figsize=(20,30))
    for i,col in enumerate(colonnes,1):
        ax = fig.add_subplot(longueur,largeur,i)
        sns.boxplot(x=df[var_comparaison],y=df[col], ax=ax, order=ordre, showfliers = outliers)
        if option:
            plt.xticks(rotation=90, ha='right')
    fig.suptitle('Boxplot de chaque target en fonction de {}'.format(var_comparaison))
    plt.tight_layout(pad = 4)
    plt.show()


    
