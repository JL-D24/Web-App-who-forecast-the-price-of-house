Project : Development of application on streamlit who forecasting price of house 

Objectives :  
                -Identify paterns and trends to predict the price of house on the Real Estate at Boston(USA).
                -Develop a model to forecast de manière automatique la valeur médiane des logements occupés par leur propriétaire.
                -Evaluate the model's perfomance
                -Implement in the app the model forecasting

Dataset
       source : https://www.kaggle.com/datasets/altavish/boston-housing-dataset
       
        size : The Boston housing dataset contains 506 observations and 14 variables

Description variables :

CRIM : Taux de criminalité par habitant par ville.

ZN : Proportion de terrains résidentiels zonés pour des lots de plus de 25 000 pieds carrés.

INDUS : Proportion de surfaces commerciales non détaillantes par ville.

CHAS : Variable binaire indiquant si le logement est situé près de la rivière Charles (1 = près, 0 = loin).

NOX : Concentration d'oxydes nitriques (pollution de l'air).

RM : Nombre moyen de pièces par logement.

AGE : Proportion de logements occupés par leur propriétaire construits avant 1940.

DIS : Distance pondérée jusqu'à cinq centres d'emploi de Boston.

RAD : Indice d'accessibilité aux autoroutes.

TAX : Taux d'imposition foncière pour 100 000 dollars de valeur.

PTRATIO : Ratio élèves/enseignants par ville.

B : Proportion de personnes afro-américaines par ville, calculée par la formule 

LSTAT : Pourcentage de la population considérée comme ayant un statut socio-économique faible.

MEDV : Valeur médiane des logements occupés par leur propriétaire, en milliers de dollars.



Résultat : Les trois modèles ont des performances très intéressantes, cependant le modèle linéaire avec son prix de prédiction le plus élevé en termes de volume n’arrive pas à bien mesurer l’écart moyen les prédictions et les données réelles avec le MAE le plus élevé des trois modèles. Donc, le meilleur modèle dans le cas de notre projet qui nous permet de mieux prédire le prix des logements sur le marché à Boston est le gradient Boosting, car son R2 le plus proche de 1 explique à plus de 89% la variance des données mais aussi minimise mieux l’écart moyen les prédictions et les données réelles(MAE). A noter aussi, le statut socio-économique des gens et le nombre de pièce dans l'appartement ont un impact considérable sur le prix des logements.
