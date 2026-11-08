# Genera el notebook del taller "Combinando Modelos: Hard vs Soft Voting" para Yomi Level.
# Uso: python build_nb.py  ->  Taller_Voting_Yomi_Level.ipynb (sin ejecutar)
import nbformat as nbf
import textos as T

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(T.INTRO)

code(r'''
# ============================================================
# 📦 Librerías
# ============================================================
import os, itertools, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report, roc_auc_score)
warnings.filterwarnings('ignore')
os.makedirs('figuras', exist_ok=True)
plt.rcParams['figure.dpi'] = 110
ORDEN = ["Fácil", "Intermedio", "Difícil"]
''')

md(T.FASE1)
code(r'''
# ============================================================
# 📂 Fase 1 · Cargar y preparar el dataset (mismo preprocesamiento del proyecto)
# ============================================================
# El CSV se lee junto al notebook; en Colab, si no se subió, se descarga del repositorio
ruta = ('moe_db_anilist_dataset.csv' if os.path.exists('moe_db_anilist_dataset.csv')
        else 'https://raw.githubusercontent.com/JoSker51/TallerVoting/main/moe_db_anilist_dataset.csv')
df = pd.read_csv(ruta)
print("Filas originales:", len(df))

# Quitamos los títulos sin ningún metadato (no tienen formato ni géneros)
df = df[df['entry_type'].notna()].copy()

# Variable objetivo: dificultad (1-100) cortada en terciles -> 3 clases del mismo tamaño
df['nivel'], cortes = pd.qcut(df['Overall Difficulty (1-100)'], 3, labels=ORDEN, retbins=True)

# Episodios y año faltantes: mediana de su mismo formato (la global no sirve para películas y series a la vez)
for c in ['episodes', 'start_year']:
    df[c] = df[c].fillna(df.groupby('entry_type')[c].transform('median'))

# Formato -> dos binarias; géneros -> multi-hot (21 géneros con al menos 100 títulos)
df['genero_desconocido'] = df['genres'].isna().astype(int)
df['es_anime'] = df['entry_type'].str.startswith('anime').astype(int)
df['es_pelicula'] = df['entry_type'].str.endswith('movie').astype(int)
g = (df['genres'].fillna('')
     .str.replace('Sci-Fi & Fantasy', 'Sci-Fi, Fantasy', regex=False)
     .str.replace('Action & Adventure', 'Action, Adventure', regex=False)
     .str.replace('War & Politics', 'War', regex=False)
     .str.replace('Science Fiction', 'Sci-Fi', regex=False))
gen = g.str.get_dummies(sep=', ')
gen = gen.loc[:, gen.sum() >= 100]
gen.columns = ['g_' + c for c in gen.columns]
df = pd.concat([df, gen], axis=1)

# Logaritmo a las variables muy sesgadas
for c in ['Avg. Words/Episode', 'Unique Words', 'episodes']:
    df['log_' + c] = np.log1p(df[c])

# 28 variables. NO usamos Kanji/Vocab Difficulty ni Vocab Density: la etiqueta se calcula con ellas (fuga de datos)
NUM = ['log_Avg. Words/Episode', 'log_Unique Words', 'log_episodes', 'start_year']
FMT = ['es_anime', 'es_pelicula', 'genero_desconocido']
features = NUM + FMT + list(gen.columns)
X, y = df[features], df['nivel'].astype(str)

# Misma división de todos los talleres: 80/20 estratificada, semilla 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Títulos usados: {len(df)} | variables: {len(features)} | train: {len(X_train)} | test: {len(X_test)}")
print("Cortes de dificultad:", np.round(cortes, 1))
print("\nBalance de clases:"); print(y.value_counts().reindex(ORDEN).to_string())
''')

code(r'''
# 📊 Distribución de la dificultad y balance de clases
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
sns.histplot(df['Overall Difficulty (1-100)'], bins=60, color='gray', ax=ax[0])
for c in cortes[1:3]:
    ax[0].axvline(c, color='crimson', linestyle='--')
ax[0].set_title('Dificultad (1-100) y cortes de los terciles'); ax[0].set_xlabel('Overall Difficulty')
ax[0].set_xlim(0, 60)
conteo = y.value_counts().reindex(ORDEN)
ax[1].bar(ORDEN, conteo.values, color=['#2E7D32', '#F2A900', '#C62828'])
for i, v in enumerate(conteo.values):
    ax[1].text(i, v + 40, f"{v:,}".replace(',', '.'), ha='center')
ax[1].set_title('Balance de clases'); ax[1].set_ylabel('Títulos')
plt.tight_layout(); plt.savefig('figuras/1_dataset.png'); plt.show()
''')

md(T.FASE2)
code(r'''
# ============================================================
# 🤖 Fase 2 · Modelos base: los mismos parámetros que encontramos en cada taller
# ============================================================
# KNN y SVM necesitan escalado -> van dentro de un Pipeline con StandardScaler.
# Random Forest no lo necesita.
modelos_base = {
    # Taller KNN: GridSearchCV con 5 folds eligió K=39, distancia Manhattan y pesos por distancia
    'KNN': make_pipeline(StandardScaler(),
                         KNeighborsClassifier(n_neighbors=39, metric='manhattan', weights='distance')),
    # Taller SVM: GridSearch de 20 combinaciones eligió kernel RBF, C=100, gamma='scale'.
    # probability=True es obligatorio para poder usar Soft Voting.
    'SVM': make_pipeline(StandardScaler(),
                         SVC(kernel='rbf', C=100, gamma='scale', probability=True, random_state=42)),
    # Proyecto (modelo final): 300 árboles, profundidad 15, 40% de las variables por división
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=15, max_features=0.4,
                                            random_state=42, n_jobs=-1),
}

def errores_graves(real, pred):
    """% de títulos donde se confunde un extremo con el otro (Fácil <-> Difícil)."""
    real, pred = np.asarray(real), np.asarray(pred)
    return np.mean(((real == 'Fácil') & (pred == 'Difícil')) | ((real == 'Difícil') & (pred == 'Fácil')))

def metricas(real, pred):
    return {'Accuracy': accuracy_score(real, pred),
            'F1-macro': f1_score(real, pred, average='macro'),
            'Recall Difícil': recall_score(real, pred, labels=['Difícil'], average='macro'),
            'Errores graves': errores_graves(real, pred)}

pred, proba, tabla = {}, {}, {}
for nombre, modelo in modelos_base.items():
    modelo.fit(X_train, y_train)
    pred[nombre] = modelo.predict(X_test)
    proba[nombre] = modelo.predict_proba(X_test)
    tabla[nombre] = metricas(y_test, pred[nombre])

pd.DataFrame(tabla).T.round(3)
''')

code(r'''
# 🔲 Matrices de confusión de los modelos base
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, nombre in zip(axes, modelos_base):
    cm = confusion_matrix(y_test, pred[nombre], labels=ORDEN)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, xticklabels=ORDEN, yticklabels=ORDEN)
    ax.set_title(f"{nombre} · F1 {tabla[nombre]['F1-macro']:.3f}")
    ax.set_xlabel('Predicción'); ax.set_ylabel('Real')
plt.tight_layout(); plt.savefig('figuras/2_cm_base.png'); plt.show()
''')

md(T.FASE3)
code(r'''
# ============================================================
# 🗳️ Fase 3 · VotingClassifier: Hard y Soft con los 3 modelos base
# ============================================================
estimadores = list(modelos_base.items())

# Hard: cada modelo vota por una clase y gana la mayoría
voting_hard = VotingClassifier(estimators=estimadores, voting='hard', n_jobs=-1)
# Soft: se promedian las probabilidades de los 3 modelos y gana la clase con mayor promedio
voting_soft = VotingClassifier(estimators=estimadores, voting='soft', n_jobs=-1)

for nombre, modelo in [('Voting Hard', voting_hard), ('Voting Soft', voting_soft)]:
    modelo.fit(X_train, y_train)
    pred[nombre] = modelo.predict(X_test)
    tabla[nombre] = metricas(y_test, pred[nombre])
proba['Voting Soft'] = voting_soft.predict_proba(X_test)

resultados = pd.DataFrame(tabla).T
resultados.round(3)
''')

code(r'''
# 📋 Reporte por clase del mejor ensemble (Soft)
print(classification_report(y_test, pred['Voting Soft'], labels=ORDEN, digits=3))
''')

code(r'''
# 🔍 ¿Qué tanto coinciden los modelos entre sí? (la diversidad es lo que hace funcionar el voting)
votos = pd.DataFrame({k: pred[k] for k in modelos_base}, index=X_test.index)
print(f"Los 3 modelos dicen lo mismo en el {(votos.nunique(axis=1) == 1).mean():.1%} de los títulos")
print(f"Los 3 dicen algo distinto (empate) en el {(votos.nunique(axis=1) == 3).mean():.1%}")
for a, b in itertools.combinations(modelos_base, 2):
    print(f"  {a} y {b} coinciden en el {(votos[a] == votos[b]).mean():.1%}")
print("\n¿Con qué modelo base coincide más cada ensemble?")
for ens in ['Voting Hard', 'Voting Soft']:
    print(f"  {ens}: " + ", ".join(f"{k} {np.mean(pred[ens] == pred[k]):.1%}" for k in modelos_base))
''')

code(r'''
# 🎬 La decisión del ensemble en casos reales: títulos donde los modelos no se ponen de acuerdo
clases = list(voting_soft.classes_)
info = df.loc[X_test.index, ['Anime Title', 'entry_type', 'start_year', 'Overall Difficulty (1-100)']]
filas = []
for i, idx in enumerate(X_test.index):
    if votos.loc[idx].nunique() > 1:
        fila = {'Título': info.loc[idx, 'Anime Title'], 'Real': y_test.loc[idx]}
        for k in modelos_base:
            p = proba[k][i][clases.index(pred[k][i])]
            fila[k] = f"{pred[k][i]} ({p:.0%})"
        fila['Hard'] = pred['Voting Hard'][i]
        fila['Soft'] = f"{pred['Voting Soft'][i]} ({proba['Voting Soft'][i].max():.0%})"
        filas.append(fila)
casos = pd.DataFrame(filas)
# Casos donde Hard y Soft deciden distinto: ahí se ve el efecto de la "confianza" de cada modelo
distintos = casos[casos['Hard'] != casos['Soft'].str.split(' ').str[0]]
print(f"Hard y Soft deciden distinto en {len(distintos)} títulos ({len(distintos)/len(X_test):.1%} del test)")
print(f"Soft acierta en {np.mean(distintos['Soft'].str.split(' ').str[0] == distintos['Real']):.0%} de esos casos; "
      f"Hard en {np.mean(distintos['Hard'] == distintos['Real']):.0%}")
distintos.head(8)
''')

md(T.FASE4)
code(r'''
# ============================================================
# 📈 Fase 4 · Comparación de métricas entre modelos base y ensembles
# ============================================================
orden_m = ['KNN', 'SVM', 'Random Forest', 'Voting Hard', 'Voting Soft']
colores = ['#9AA3AD', '#7C93B0', '#2B4A6F', '#F2A900', '#D9480F']
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, met, lim in zip(axes, ['Accuracy', 'F1-macro', 'Errores graves'], [(0.50, 0.62), (0.50, 0.62), (0, 0.09)]):
    v = resultados.loc[orden_m, met]
    barras = ax.bar(orden_m, v, color=colores)
    for b in barras:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (lim[1] - lim[0]) * 0.015,
                f"{b.get_height():.1%}" if met == 'Errores graves' else f"{b.get_height():.3f}", ha='center', fontsize=9)
    ax.set_ylim(*lim); ax.set_title(met + (' (menos es mejor)' if met == 'Errores graves' else ''))
    ax.tick_params(axis='x', rotation=30); ax.grid(axis='y', linestyle='--', alpha=0.6)
axes[0].axhline(1/3, color='gray', linestyle=':')
plt.tight_layout(); plt.savefig('figuras/3_comparacion.png'); plt.show()
''')

code(r'''
# 🔲 Matrices de confusión: Hard vs Soft
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, nombre in zip(axes, ['Voting Hard', 'Voting Soft']):
    cm = confusion_matrix(y_test, pred[nombre], labels=ORDEN)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges' if 'Soft' in nombre else 'YlOrBr', cbar=False,
                ax=ax, xticklabels=ORDEN, yticklabels=ORDEN)
    ax.set_title(f"Matriz de confusión · {nombre}"); ax.set_xlabel('Predicción'); ax.set_ylabel('Real')
plt.tight_layout(); plt.savefig('figuras/4_cm_voting.png'); plt.show()
''')

code(r'''
# ➕➖ ¿Cómo cambia el desempeño al quitar o agregar modelos del ensemble?
# Combinaciones de 2 y 3 modelos, en Hard y en Soft
ablacion = []
for r in (1, 2, 3):
    for combo in itertools.combinations(modelos_base, r):
        for tipo in (['hard', 'soft'] if r > 1 else ['individual']):
            if r == 1:
                p = pred[combo[0]]
            else:
                vc = VotingClassifier([(k, modelos_base[k]) for k in combo], voting=tipo, n_jobs=-1).fit(X_train, y_train)
                p = vc.predict(X_test)
            ablacion.append({'Modelos': ' + '.join(combo), 'n': r, 'Voting': tipo,
                             'F1-macro': f1_score(y_test, p, average='macro'), 'Errores graves': errores_graves(y_test, p)})
ablacion = pd.DataFrame(ablacion)
ablacion.round(4)
''')

code(r'''
# 📊 Gráfica de la ablación
fig, ax = plt.subplots(figsize=(11, 5))
ind = ablacion[ablacion['n'] == 1].set_index('Modelos')['F1-macro']
comb = ablacion[ablacion['n'] > 1].pivot(index='Modelos', columns='Voting', values='F1-macro')
comb = comb.loc[['KNN + SVM', 'KNN + Random Forest', 'SVM + Random Forest', 'KNN + SVM + Random Forest']]
etiquetas = list(ind.index) + list(comb.index)
xs = np.arange(len(etiquetas))
ax.bar(xs[:3], ind.values, color='#9AA3AD', width=0.55, label='Modelo individual')
ax.bar(xs[3:] - 0.18, comb['hard'].values, width=0.36, color='#F2A900', label='Hard Voting')
ax.bar(xs[3:] + 0.18, comb['soft'].values, width=0.36, color='#D9480F', label='Soft Voting')
for x, v in zip(xs[:3], ind.values): ax.text(x, v + 0.002, f"{v:.3f}", ha='center', fontsize=9)
for x, h, s in zip(xs[3:], comb['hard'].values, comb['soft'].values):
    ax.text(x - 0.18, h + 0.002, f"{h:.3f}", ha='center', fontsize=8); ax.text(x + 0.18, s + 0.002, f"{s:.3f}", ha='center', fontsize=8)
ax.axhline(ind.max(), color='#2B4A6F', linestyle='--', linewidth=1, label='Mejor individual (Random Forest)')
ax.set_xticks(xs); ax.set_xticklabels(etiquetas, rotation=20, ha='right'); ax.set_ylim(0.48, 0.625)
ax.set_ylabel('F1-macro'); ax.set_title('F1-macro al quitar o agregar modelos del ensemble')
ax.legend(loc='upper left', fontsize=9); ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout(); plt.savefig('figuras/5_ablacion.png'); plt.show()
''')

md(T.DESBALANCE_INTRO)
code(r'''
# ============================================================
# ⚖️ Experimento extra: problema desbalanceado "¿Es Difícil? (sí/no)" -> 1 de cada 3 es positivo
# ============================================================
ytr_b, yte_b = (y_train == 'Difícil').astype(int), (y_test == 'Difícil').astype(int)
print(f"Proporción de títulos difíciles en test: {yte_b.mean():.1%}")

from sklearn.base import clone
bin_modelos = {k: clone(m) for k, m in modelos_base.items()}
bin_res, bin_proba = {}, {}
for k, m in bin_modelos.items():
    m.fit(X_train, ytr_b); p = m.predict(X_test); bin_proba[k] = m.predict_proba(X_test)[:, 1]; bin_res[k] = p
for tipo in ['hard', 'soft']:
    vc = VotingClassifier(list(bin_modelos.items()), voting=tipo, n_jobs=-1).fit(X_train, ytr_b)
    bin_res[f'Voting {tipo.title()}'] = vc.predict(X_test)
    if tipo == 'soft':
        bin_proba['Voting Soft'] = vc.predict_proba(X_test)[:, 1]

def fila_bin(real, p, pr=None):
    tn, fp, fn, tp = confusion_matrix(real, p).ravel()
    return {'Accuracy': accuracy_score(real, p), 'F1 (Difícil)': f1_score(real, p),
            'Precisión': precision_score(real, p), 'Recall': recall_score(real, p),
            'Falsos positivos': fp, 'Falsos negativos': fn,
            'AUC': roc_auc_score(real, pr) if pr is not None else np.nan}

tabla_bin = pd.DataFrame({k: fila_bin(yte_b, p, bin_proba.get(k)) for k, p in bin_res.items()}).T
tabla_bin.round(3)
''')

code(r'''
# 🎚️ Soft Voting permite mover el umbral: ¿cuántos títulos difíciles detecta si somos más exigentes?
umbrales = np.arange(0.20, 0.61, 0.05)
curva = pd.DataFrame([fila_bin(yte_b, (bin_proba['Voting Soft'] >= u).astype(int)) | {'Umbral': round(u, 2)}
                      for u in umbrales]).set_index('Umbral')
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(curva.index, curva['Recall'], marker='o', label='Recall (difíciles detectados)', color='#C62828')
ax[0].plot(curva.index, curva['Precisión'], marker='s', label='Precisión', color='#2B4A6F')
ax[0].plot(curva.index, curva['F1 (Difícil)'], marker='d', label='F1', color='#D9480F')
ax[0].axvline(0.5, color='gray', linestyle=':'); ax[0].set_xlabel('Umbral de probabilidad'); ax[0].legend(); ax[0].grid(alpha=0.5)
ax[0].set_title('Soft Voting: efecto del umbral')
fpfn = tabla_bin[['Falsos positivos', 'Falsos negativos']].loc[['KNN', 'SVM', 'Random Forest', 'Voting Hard', 'Voting Soft']]
fpfn.plot(kind='bar', ax=ax[1], color=['#F2A900', '#C62828'], rot=25)
ax[1].set_title('Falsos positivos vs falsos negativos (umbral 0,5)'); ax[1].grid(axis='y', alpha=0.5)
plt.tight_layout(); plt.savefig('figuras/6_desbalanceado.png'); plt.show()
curva.round(3)
''')

md(T.FASE5)
md(T.CONCLUSION)

nb = nbf.v4.new_notebook()
nb['cells'] = C
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                  'language_info': {'name': 'python'}}
nbf.write(nb, 'Taller_Voting_Yomi_Level.ipynb')
print('notebook generado')
