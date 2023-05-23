# <center>MACHINE LEARNING</center>

<details close>

<summary align='center'> <h2> ML00 : Matrice, prédiction et évaluation </h2> </summary>

### <center>Les matrices</center>

La matrice est un moyen pour ordonner des donnes et simplifier leurs utilisations. Par exemple, grâce à des méthodes qui existent pour appliquer une même opération à plusieurs données simultanément. Elles sont généralement organisées de manière à avoir les données pour une prédiction par ligne.

### <center>Prédiction Linéaire (du premier degré)</center>

Une prédiction linéaire et le résultat de l'opération _`ax + b = ŷ`_ où _a_ et _b_ sont des constantes et _x_ une valeur utilise pour prédire un _ŷ_ associé. Le but du machine learning (avec une prédiction linéaire) est d'avoir pour chaque _x_ un _ŷ_ le plus proche de la valeur attendue _y_, pour pouvoir reproduire l'operation sur des _x_ dont le _y_ est inconnu.

_a_ et _b_ sont stockés dans une matrice que l'on nommera _θ_.

Pour prédire aisément les _ŷ_ d'un jeu de donner noter _X_ nous pouvons réaliser l'opération _`X·θ = Ŷ`_

### <center>Calculer la perte</center>

Calculer la perte permet de savoir à quel point notre modèle est éloigné de la réalité, plus cette valeur est proche de 0 plus le résultat est proche de la vérité.

On peut calculer la perte de chaque élément en soustrayant la valeur obtenue avec la valeur attendue et en passant le résultat en positif, par exemple, la mettant au carré ou en prenant sa valeur absolue.

On peut aussi utiliser des fonctions permettant, grâce à la perte par élément, de déterminer la précision du modèle actuel. Ces fonctions sont très variées, il existe par exemple la MSE, la RMSE, la MAE ou la R2score.

</details>
