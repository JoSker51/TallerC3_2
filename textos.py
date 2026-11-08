# Textos (markdown) del notebook y del mini informe. Todos los números salen de la ejecución del notebook.

INTRO = r'''
# Taller: Combinando Modelos · Hard vs Soft Voting en Yomi Level
**Aprendizaje Supervisado · Universidad Sergio Arboleda**
José Santiago González · Oscar Mateo Arrubla · Leonardo Diasgranados · Docente: Hernán Cruz Bueno

**Sector:** aprendizaje de japonés con contenido audiovisual (anime, doramas y cine japonés).
**Pregunta:** ¿combinar los modelos que ya entrenamos en los talleres (KNN, SVM y Random Forest) predice mejor si el japonés de un título será *Fácil*, *Intermedio* o *Difícil* para un estudiante?
'''

FASE1 = r'''
## Fase 1 · Exploración del dataset

**Dataset:** Moe-DB + AniList + TMDB, con **9.977 títulos** japoneses (anime de TV, películas de anime, doramas y películas live-action). Tras quitar 943 títulos sin metadatos quedan **9.034**.

| Aspecto | Detalle |
|---|---|
| Variables de entrada | **28**: palabras por episodio, palabras únicas y episodios (en logaritmo), año de estreno, anime/live-action, película/serie, género desconocido y 21 géneros (multi-hot) |
| Variable objetivo | `Overall Difficulty` (1-100) cortada en terciles: **Fácil** (< 19,9), **Intermedio** (19,9-25,4) y **Difícil** (> 25,4) |
| Tipo de salida | Categórica, 3 clases → **clasificación multiclase** |
| Tamaño | 7.227 títulos de entrenamiento y 1.807 de prueba (80/20 estratificado, semilla 42) |
| Balance | **Balanceado**: 3.017 / 3.015 / 3.002. Adivinar al azar acierta 33% |

**¿Por qué clasificación y no regresión?** La decisión del negocio es por niveles: la app muestra un semáforo (verde, amarillo, rojo) y recomienda según el nivel del estudiante. Además, la dificultad exacta está muy concentrada entre 15 y 30 puntos (gráfica de abajo), así que predecir "23,4 frente a 24,1" no cambia ninguna decisión; lo útil es saber en qué grupo cae. Es la misma variable objetivo de todos nuestros talleres, lo que permite comparar el ensemble con los modelos que ya teníamos.

**Fuga de datos evitada:** no usamos la dificultad de kanji, la de vocabulario ni la densidad de vocabulario, porque `Overall Difficulty` se calcula con ellas.
'''

FASE2 = r'''
## Fase 2 · Modelos base

Usamos los **3 modelos que trabajamos en los talleres**, cada uno con los parámetros que encontramos allí. Son de **familias distintas**, que es lo que pide el voting:

| Modelo | Familia | Parámetros (del taller) | Cómo decide |
|---|---|---|---|
| **KNN** | Basado en distancias | K=39, Manhattan, pesos por distancia (GridSearchCV, 5 folds) | Vota entre los 39 títulos más parecidos |
| **SVM** | Basado en márgenes | Kernel RBF, C=100, gamma='scale' (GridSearch de 20 combinaciones) | Traza la frontera con mayor margen entre clases |
| **Random Forest** | Ensamble de árboles | 300 árboles, profundidad 15, 40% de variables por división | 300 árboles con reglas "si… entonces" que votan |

Medimos **Accuracy**, **F1-macro**, el **recall de la clase Difícil** (la que más importa: recomendar algo difícil a un principiante provoca abandono) y los **errores graves** (confundir Fácil con Difícil o al revés).
'''

FASE3 = r'''
## Fase 3 · Voting Ensemble

- **Hard Voting:** cada modelo vota por un nivel y gana la mayoría. Si los 3 dicen algo distinto, hay empate.
- **Soft Voting:** cada modelo da una probabilidad para cada nivel, se promedian y gana el nivel con mayor promedio. Por eso el SVM necesita `probability=True`.
'''

FASE4 = r'''
## Fase 4 · Visualización

Primero comparamos las métricas y las matrices de confusión; después medimos qué pasa al **quitar o agregar modelos** del ensemble (ablación).
'''

DESBALANCE_INTRO = r'''
### Experimento extra: ¿y si el problema fuera desbalanceado?

Nuestras 3 clases están balanceadas por construcción. Para responder la pregunta de la Fase 5 sobre datos **desbalanceados**, repetimos el ejercicio con la pregunta del negocio en forma binaria: **"¿este título es Difícil, sí o no?"**. Ahí solo 1 de cada 3 títulos es positivo. En este caso un **falso negativo** es un título difícil que el sistema deja pasar como "no difícil" (llega a un principiante), y un **falso positivo** es un título que no era difícil pero se marca en rojo (se pierde una recomendación).
'''

FASE5 = r'''
## Fase 3 (preguntas) y Fase 5 · Análisis y reflexión

### ¿El ensemble mejora el rendimiento global?
**Sí, el Soft Voting mejora a los 3 modelos base en todas las métricas globales**, aunque la mejora es moderada:

| Modelo | Accuracy | F1-macro | Recall Difícil | Errores graves |
|---|---|---|---|---|
| KNN | 0,546 | 0,544 | 0,602 | 7,9% |
| SVM | 0,570 | 0,568 | 0,662 | 6,5% |
| Random Forest | 0,581 | 0,580 | 0,650 | 6,1% |
| Voting Hard | 0,583 | 0,580 | **0,673** | 6,8% |
| **Voting Soft** | **0,593** | **0,591** | 0,663 | **5,6%** |

El Soft Voting acierta **59,3 de cada 100 títulos** frente a 58,1 del mejor modelo solo (Random Forest), y baja los errores graves de 6,1% a **5,6%**. El Hard Voting, en cambio, **casi empata con Random Forest** (F1 0,580 en ambos) y comete más errores graves.

### ¿Qué tipo fue mejor, Hard o Soft, y por qué?
**Soft Voting**, por tres razones que se ven en los datos:
1. **Aprovecha la confianza de cada modelo.** Hard y Soft deciden distinto en **149 títulos (8,2% del test)**, y en esos casos Soft acierta el **44%** frente al **31%** de Hard. Ejemplo: en *Ginban Kaleidoscope* (realmente Intermedio) KNN dice Intermedio con 50%, SVM Difícil con 41% y Random Forest Fácil con 43%. Hard rompe el empate a ciegas y dice Difícil; Soft promedia y elige Intermedio, que es lo correcto.
2. **Nuestros modelos dudan mucho.** En la mitad de los títulos, la probabilidad que cada modelo le da a su nivel elegido es menor al **52-57%** (mediana: KNN 54%, SVM 52%, Random Forest 57%), porque muchos títulos están cerca de los límites entre niveles. Hard convierte un "Difícil al 41%" en un voto completo, igual que un "Difícil al 90%"; Soft no.
3. **Con 3 clases hay empates.** En el 3,2% de los títulos cada modelo dice algo distinto, y con 2 modelos los empates son mucho más frecuentes. En un empate, `VotingClassifier` en modo hard elige la primera clase en orden alfabético (**"Difícil"**). Por eso Hard con dos modelos dispara los errores graves hasta **8,9%** (KNN + SVM). Soft nunca necesita desempatar así.

### ¿Qué modelo individual influye más en el resultado final?
**El SVM.** Lo vemos de dos formas:
- **Coincidencia:** el Soft Voting coincide con el SVM en el **85,0%** de los títulos, más que con Random Forest (83,2%) o KNN (82,6%). El Hard coincide también más con el SVM (86,8%).
- **Ablación (quitar un modelo del Soft):** sin el SVM el F1 cae de 0,591 a **0,572** (la mayor caída); sin Random Forest, a 0,581; sin KNN, a 0,584.

Aunque Random Forest es el mejor modelo solo, el SVM es el que más **aporta algo distinto** al grupo: es el que más detecta títulos difíciles (recall 0,662) y comete pocos errores graves.

### ¿Cuál fue el modelo base más fuerte y cuál el más débil? ¿Por qué?
- **Más fuerte: Random Forest** (F1 0,580, errores graves 6,1%). Elige solo las variables útiles en cada división, combina sin problema números y columnas de sí/no, y al promediar 300 árboles reduce la varianza.
- **Más débil: KNN** (F1 0,544, errores graves 7,9%). Mide distancias tratando las 28 variables casi por igual, así que los 21 géneros pesan tanto como el vocabulario, que es lo que más explica la dificultad. Con tantas columnas los "vecinos" se vuelven menos parecidos (maldición de la dimensionalidad).

### ¿El ensemble mejoró respecto a todos los modelos base? ¿En qué métricas?
- **Soft Voting:** sí en **Accuracy, F1-macro y errores graves**. En recall de Difícil queda igual al SVM (0,663 frente a 0,662). En el problema binario además tiene el **mejor AUC (0,831** frente a 0,820 de Random Forest), es decir, es el que mejor ordena los títulos por dificultad.
- **Hard Voting:** mejora en **recall de Difícil (0,673, el más alto)** y apenas en Accuracy (0,583 frente a 0,581). En F1 empata con Random Forest y en errores graves es peor que SVM y Random Forest.

### ¿Qué conclusiones sacamos al comparar Hard y Soft?
- **Soft le gana a Hard en las 4 combinaciones posibles de modelos** (gráfica de ablación), y la diferencia es mayor con 2 modelos, donde Hard sufre los empates.
- **Hard solo vale la pena con muchos modelos parecidos en calidad** o cuando los modelos no dan probabilidades confiables. Con 3 clases y modelos que dudan, se pierde información.
- **Soft cuesta más:** el SVM necesita `probability=True`, que hace una validación cruzada interna y multiplica el tiempo de entrenamiento.

### ¿En qué tipo de dataset, balanceado o desbalanceado, se comporta mejor Soft Voting?
**En el desbalanceado es donde Soft muestra su mayor ventaja, siempre que se ajuste el umbral.** En el problema binario "¿es Difícil?" (33% de positivos):
- Con el umbral por defecto (0,5), Soft es **conservador**: tiene la mejor precisión (0,755) pero solo detecta el 49,8% de los difíciles, porque al promediar, las probabilidades de la clase minoritaria rara vez pasan de 0,5.
- Pero como Soft entrega una **probabilidad**, se puede bajar el umbral. Con **0,35**, detecta el **72,8%** de los títulos difíciles con F1 **0,673**, el mejor de todos los modelos. Hard no tiene esa opción: solo da votos.

En el dataset balanceado la ventaja de Soft existe (+0,011 de F1) pero es pequeña, porque ninguna clase queda "aplastada" por las demás.

### Si quitáramos uno de los modelos, ¿cambiaría mucho el resultado?
**Depende de cuál, pero ningún cambio es dramático en Soft Voting:**

| Ensemble Soft | F1-macro | Errores graves |
|---|---|---|
| Los 3 (KNN + SVM + RF) | **0,591** | **5,6%** |
| Sin KNN (SVM + RF) | 0,584 | 5,7% |
| Sin RF (KNN + SVM) | 0,581 | 6,5% |
| Sin SVM (KNN + RF) | 0,572 | 6,6% |

Quitar **KNN** casi no se nota: es el modelo más débil y buena parte de lo que aporta ya lo cubre el SVM (coinciden en el 74% de los títulos). Quitar el **SVM** es lo que más daña. En **Hard Voting** quitar cualquier modelo sí cambia mucho: el F1 cae a 0,55-0,56 por los empates de dos votantes.

### ¿Qué importancia tiene que los modelos base sean diversos?
**Es la razón por la que el ensemble funciona.** Los 3 modelos dicen lo mismo solo en el **60,1%** de los títulos. Cada par coincide entre **71% y 74%**, así que se equivocan en títulos distintos y unos corrigen a otros. KNN decide por parecido, SVM por una frontera y Random Forest por reglas, así que miran el problema de formas distintas. Si usáramos 3 Random Forest con distinta semilla, coincidirían casi siempre y el voto no agregaría nada nuevo.

### ¿Qué tipo de error se redujo más con el ensemble?
- **En las 3 clases:** se redujeron los **errores graves**, sobre todo **Fácil → Difícil**. Ese error pasa de 58 (KNN), 62 (SVM) y 48 (RF) a **46** con Soft. **Difícil → Fácil** queda en 56, cerca del mejor modelo base (SVM, 55).
- **En el problema binario (umbral 0,5): se redujeron los falsos positivos.** Soft tiene **97** falsos positivos, frente a 124-161 de los modelos base. Los **falsos negativos no se redujeron** (301, más que SVM con 246 y RF con 269).

En nuestro sector el **falso negativo** es el error caro: un título difícil que llega a un principiante. Por eso, si usamos el ensemble, hay que bajar el umbral (0,35-0,40). Con 0,35 los falsos negativos bajan a **163**, menos que cualquier modelo base, a cambio de más títulos marcados en rojo "por precaución".
'''

CONCLUSION = r'''
## Conclusión: ¿cuál elegiríamos y por qué?

**Elegimos el Soft Voting con KNN + SVM + Random Forest** como modelo de clasificación de Yomi Level:
- Es el mejor en **Accuracy (0,593), F1-macro (0,591) y errores graves (5,6%)**, y el que mejor ordena por dificultad (**AUC 0,831** en el problema binario).
- Su **probabilidad promedio** sirve directo para el semáforo de la app: confianza alta → color firme; confianza baja → "zona gris" para revisión humana. Y permite ajustar el umbral para **detectar más títulos difíciles**, que es lo que protege al estudiante principiante.

**Con dos reservas honestas:**
1. **La mejora es moderada** (+1,1 puntos de F1 sobre Random Forest). El límite no está en el modelo sino en la información: con 28 variables de ficha y conteos de palabras hay títulos que ningún modelo puede separar bien, sobre todo los Intermedios (F1 0,459).
2. **Es más pesado:** hay que entrenar 3 modelos, y el SVM con probabilidades es el más lento. Para la app que corre en el navegador seguiríamos usando Random Forest solo; el Soft Voting conviene en un servidor, donde el costo extra no importa.

**Hard Voting no lo elegiríamos:** empata con Random Forest en F1, comete más errores graves y, con 3 clases, sus empates se resuelven a favor de "Difícil" sin ningún criterio.
'''
