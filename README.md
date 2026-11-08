# Combinando Modelos · Hard vs Soft Voting — Yomi Level

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoSker51/TallerVoting/blob/main/Taller_Voting_Yomi_Level.ipynb)

**Aprendizaje Supervisado · Universidad Sergio Arboleda**
José Santiago González · Oscar Mateo Arrubla · Leonardo Diasgranados · Docente: Hernán Cruz Bueno

**Sector:** aprendizaje de japonés con contenido audiovisual. **Tarea:** clasificar el japonés de cada título en *Fácil*, *Intermedio* o *Difícil* (9.034 títulos, 3 clases balanceadas).

## Entregables

| Archivo | Contenido |
|---|---|
| [`Taller_Voting_Yomi_Level.ipynb`](Taller_Voting_Yomi_Level.ipynb) | Notebook ejecutado y comentado: las 5 fases del taller |
| [`Informe_Taller_Voting_Yomi_Level.pdf`](Informe_Taller_Voting_Yomi_Level.pdf) | Mini informe: dataset, modelos base, Hard y Soft Voting, reflexión y conclusión |
| [`figuras/`](figuras/) | Gráficos comparativos y matrices de confusión |

## Resultados (1.807 títulos de prueba)

| Modelo | Accuracy | F1-macro | Errores graves |
|---|---|---|---|
| KNN (K=39, Manhattan) | 0,546 | 0,544 | 7,9% |
| SVM (RBF, C=100) | 0,570 | 0,568 | 6,5% |
| Random Forest (300 árboles) | 0,581 | 0,580 | 6,1% |
| Voting Hard | 0,583 | 0,580 | 6,8% |
| **Voting Soft** | **0,593** | **0,591** | **5,6%** |

**Elegimos Soft Voting:** mejora a los 3 modelos base, reduce los errores graves (confundir Fácil con Difícil) y su probabilidad promedio permite ajustar el umbral para detectar más títulos difíciles.

## Cómo ejecutar

**En Colab:** botón "Abrir en Colab" y *Entorno de ejecución → Ejecutar todas*. El notebook descarga el dataset de este repositorio.

**En local:**

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook Taller_Voting_Yomi_Level.ipynb
```

`build_nb.py`, `textos.py` y `build_doc.py` regeneran el notebook y el informe.
