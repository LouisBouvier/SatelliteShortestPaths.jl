## Intro

Le Machine Learning et la Recherche opérationnelle appartiennent tous deux au domaine de l'IA, mais leurs approches diffèrent. Le Machine Learning tire parti des données passées pour effectuer des tâches telles que la classification ou la régression (à l'aide d'algorithmes tels que SVM, arbres de décision, CNN, etc.), tandis que la recherche opérationnelle vise à optimiser l'utilisation des ressources afin d'améliorer la prise de décision et l'efficacité (en utilisant des méthodes telles que SIMPLEX, PSO, etc.). Leurs différences dans l'approche des problèmes fait que ces deux domaines sont rarement combinés pour résoudre des problèmes.

Dans ce blog, nous allons explorer une approche novatrice pour trouver le chemin le plus court sur une image, en combinant judicieusement des techniques de Machine Learning (vision par ordinateur) et de Recherche Opérationnelle (recherche de chemin). Nous débuterons par une méthode naïve qui se concentre exclusivement sur le Machine Learning, puis nous présenterons une seconde solution qui surmonte les défis rencontrés en intégrant les deux approches.

## La problématique

### Le sujet

En tant que chef d'une équipe du rallye Paris/Dakar, notre véhicule est capable de circuler sur divers terrains, bien que sa vitesse puisse varier. Nous disposons de photos satellites de chaque étape de la course. L'année dernière, nous avons dû calculer manuellement le meilleur itinéraire en nous basant sur ces photos, ce qui nous a pris plusieurs nuits. Cette année, nous comptons sur notre informaticien pour simplifier cette tâche et déterminer les chemins les plus courts en quelques minutes seulement.

### La technologie

Le code que nous allons partager est écrit en Julia, car cette technologie répond à nos exigences en termes de performance et de concision. De plus, bien que la communauté soit encore relativement restreinte, de nombreux modules ont été développés pour le Machine Learning et la Recherche Opérationnelle (tels que Flux.jl, Graphs, etc.). Vous pouvez trouver les instructions pour le téléchargement ici. Aucune expérience préalable dans ce langage n'est nécessaire pour suivre la suite.

## Le Dataset

Le jeu de données avec lequel nous travaillerons est constitué du dossier d'images suivant. La préparation des données d'entraînement se déroulera en deux étapes, à savoir la traduction des types de sols en graphes pondérés, et la prise en compte des routes dans ce processus.

### Le type de sol

Il est évident que votre voiture ne va pas rouler aussi si elle est en forêt ou en montagne ou sur un autre sol. Il va falloir associer des poids à chacun des types de sol et pour cela il est nécessaire de savoir les classer. C'est pourquoi nous avons utiliser un premier Dataset que vous pouvez télécharger ici. Nous avons entraîner un premier modèle de Deep Learning permettant cette classification. Comme il ne s'agit pas du sujet principal nous allons partager les poids du modèle entraîné mais nous rentrerons pas dans les détails techniques. Vous pouvez néanmoins consulter le notebook permettant de préparer les données ici.

### Favoriser les routes

Nous souhaitons également distinguer les zones proches et éloignées des routes, en partant du principe que plus une zone est éloignée d'une route, plus elle est sauvage, ce qui peut rendre la progression difficile. Pour ce faire, nous utilisons les masques des routes présents dans notre jeu de données initial. Nous détaillerons le code Python permettant d'effectuer cette transformation. Pour commencer, nous chargeons les données :

Une fois les images chargées nous allons utiliser la librairie skimage pour calculer les distances maps.

Une fois les distances maps obtenues on peut les normaliser et les appliquer aux images satellites.

Il ne reste plus qu'à découper cette image en case pour définir notre graphique pondéré et moyenner les 3 canaux (RGB). Nous avons fait le choix d'utiliser des imagettes de 32x32 pour définir chaque noeuds du graphique.

On peut alors travailler combiner nos deux matrices de poids en les multipliant et obtenir le poids finaux qui seront utilisés par l'algorithme.

### Calculer les chemins les plus courts

Parfait ! Maintenant que nous avons notre matrice de poids, nous allons utiliser le langage Julia et ses bibliothèques pour d'abord définir un véritable graphe plutôt qu'un simple tableau de poids. Ensuite, nous calculerons le chemin le plus court pour aller du coin supérieur gauche au coin inférieur droit pour chaque image. Avec ces éléments en place, nous serons prêts à lancer notre algorithme.

## La solution naive


## La solution RO + ML

### L'architecture


### La théorie

Cette approche est novatrice car elle cherche à exploiter les capacités des réseaux de neurones pour apprendre le chemin le plus court sur une image, ce qui est un défi complexe. Contrairement à la plupart des tâches de Machine Learning qui reposent sur la rétropropagation popularisée par Hinton en 1986, cette approche se heurte à un obstacle majeur : les algorithmes de recherche opérationnelle ne sont pas dérivables. En d'autres termes, il est difficile d'appliquer la rétropropagation et la descente de gradient dans ce contexte, car ces méthodes nécessitent que chaque étape soit dérivable.

Pour surmonter cette difficulté, l'article propose une régularisation visant à rendre le processus dérivable, permettant ainsi l'entraînement du réseau de neurones. Il est important de comprendre que cette approche implique un compromis entre la performance du modèle et sa dérivabilité.