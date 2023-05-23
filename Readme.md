<h1 align='center'> MACHINE LEARNING POOL </h1>

>### **Fiche technique:**
#### &emsp;&emsp;&bull; 42 cursus -> Hors tronc commun
#### &emsp;&emsp;&bull; Langage: Python

<details open>

<summary align='center'>  <h2> ML00 : Matrice, prédiction et évaluation </h2> </summary>

<details close>
<summary> <h4> Les matrices </h4> </summary>

Les matrices sont un moyen pour ordonner les donnes et de simplifier leur utilisation, car des méthodes existent pour appliquer une même opération à plusieurs donnes simultanément. Elles sont généralement organisées de manière à avoir les données pour la prédiction d'une valeur par ligne.

</details>

<details close>
<summary> <h4> Prédiction Linéaire (du premier degré) </h4> </summary>

Une prédiction linéaire et le résultat de l'opération _`ax + b = ŷ`_ ou _a_ et _b_ sont des constantes et _x_ une valeur utilise pour prédire un _ŷ_ associé. Le but du machine learning est d'avoir pour chaque _x_ un _ŷ_ le plus proche de la valeur attendue _y_.

_a_ et _b_ sont stockés dans une matrice que l'on nommera _θ_.

Pour prédire aisément les _ŷ_ d'un jeu de donner noter _X_ nous pouvons réaliser l'opération _`X·θ = Ŷ`_

</details>

<details close>
<summary> <h4> Calculer la perte </h4> </summary>

Calculer la perte permet de savoir à quel point notre modèle est éloigné de la réalité, plus cette valeur est proche de 0 plus le résultat est proche de la vérité.

On peut calculer la perte de chaque élément en soustrayant la valeur obtenue avec la valeur attendue et en passant le résultat en positif, par exemple, la mettant au carré ou en prenant sa valeur absolue.

On peut aussi utiliser des fonctions permettant, grâce à la perte par élément, de déterminer la précision du modèle actuel. Ces fonctions sont très variées, il existe par exemple la MSE, la RMSE, la MAE ou la R2score.

</details>

</details>
