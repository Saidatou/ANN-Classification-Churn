# ANN-Classification-Churn
J’ai construit un outil basé sur l’intelligence artificielle qui permet de prédire si un client d’une banque risque de partir. J’ai ensuite créé une application simple et intuitive pour que cette prédiction puisse être faite facilement sur streamlit.
# Contexte
Dans le secteur bancaire, comprendre pourquoi un client risque de quitter la banque est essentiel pour améliorer la satisfaction et la fidélité. Comme de nombreux clients partent chaque année sans prévenir, il est important d’anticiper ce comportement.

# Objectif
L’objectif de ce projet est de créer un outil capable de prédire si un client est susceptible de quitter la banque, à partir de ses informations (comme son âge, son ancienneté, son activité sur le compte, etc.).
Ce modèle permet à la banque d’agir en amont, en identifiant les clients à risque et en mettant en place des actions ciblées pour les retenir.

Pour cela, j’ai utilisé un modèle d’intelligence artificielle inspiré du fonctionnement du cerveau humain, qui apprend à partir de données historiques pour faire des prédictions. Ce modèle analyse des milliers de profils clients pour repérer des signes de départ potentiel.

# Rappel :

ANN (Artificial Neural Network) et RNN (Recurrent Neural Network) sont deux types de réseaux de neurones artificiels, utilisés principalement en intelligence artificielle et apprentissage automatique (machine learning). Voici une explication simple :

🔹 ANN (Artificial Neural Network)
Un réseau de neurones artificiels est un modèle informatique inspiré du fonctionnement du cerveau humain. Il est composé de neurones artificiels organisés en couches : Entrée → données d’entrée (ex : pixels d’une image). Couches cachées → traitement intermédiaire.

Sortie → prédiction ou classification finale. Utilisation : Reconnaissance d’image, Classification (spam ou non-spam), Régression (prédiction de valeur). 🧠 C’est un modèle statique : il n’a pas de mémoire du passé.

🔹 RNN (Recurrent Neural Network)
Le réseau de neurones récurrent est une extension de l’ANN, avec une mémoire. Il traite des données séquentielles (suite de données dans le temps). Chaque sortie dépend des entrées actuelles ET des entrées précédentes (mémoire interne).

Il y a des boucles dans le réseau → permet de « se souvenir » du contexte. Utilisation : Traitement du langage naturel (ex : traduction automatique, chatbot), Reconnaissance vocale.Analyse de séries temporelles (ex : prévisions boursières).


# Étapes de mise en place du projet (version vulgarisée)
1) Préparer l’environnement de travail
J’ai commencé par créer un espace de travail propre et isolé, qui me permet de travailler sur le projet sans interférer avec d’autres programmes installés sur l’ordinateur.

2) Activer l’environnement
Une fois cet espace prêt, je l’ai activé pour que tous les outils et bibliothèques utilisés soient bien organisés à l’intérieur.

3) Installer les outils nécessaires
J’ai installé tous les outils (appelés bibliothèques) nécessaires au fonctionnement du projet, comme ceux qui permettent de manipuler les données ou de créer des modèles d’intelligence artificielle.

4) Préparer un outil de travail interactif (Jupyter Notebook)
J’ai mis en place un carnet interactif qui me permet d’écrire du code, de tester mes idées et de visualiser les résultats facilement, étape par étape.

5) Construire et sauvegarder le modèle prédictif
J’ai créé un fichier de travail dans lequel j’ai :

développé le modèle d’intelligence artificielle qui prédit si un client risque de partir, transformé certaines informations (comme le pays ou le genre) pour qu’elles soient compréhensibles par le modèle,

standardisé les données (pour les mettre sur la même échelle), et enregistré tout cela pour pouvoir le réutiliser plus tard sans avoir à tout reconstruire.

6) Créer un fichier de test pour la prédiction
J’ai ensuite conçu un fichier qui me permet de tester le modèle avec de nouveaux exemples de clients, pour voir s’il prédit correctement leur départ ou non.

7) Développer une interface utilisateur (avec Streamlit)
Pour rendre le modèle accessible à tout le monde, j’ai créé une petite application web. Elle permet à un utilisateur de saisir les informations d’un client et d’obtenir instantanément une prédiction sur son départ possible.

8) Lancer l’application en local
J’ai lancé l’application sur mon ordinateur pour tester son bon fonctionnement en conditions réelles.

9) Partager le projet en ligne (GitHub)
J’ai publié le code complet du projet sur une plateforme collaborative (GitHub), pour le conserver, le versionner, ou le partager avec d’autres.

10) Mettre l’application en ligne (Streamlit Cloud)
Enfin, j’ai déployé l’application sur internet pour qu’elle soit accessible depuis n’importe quel navigateur, sans avoir besoin d’installer quoi que ce soit.
