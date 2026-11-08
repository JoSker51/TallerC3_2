# Genera el documento Word del taller de Voting a partir de los mismos textos y figuras del notebook.
# Requiere haber ejecutado antes el notebook (crea la carpeta figuras/).
import re
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import textos as T

doc = Document()
st = doc.styles['Normal']; st.font.name = 'Calibri'; st.font.size = Pt(11)
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.2); s.top_margin = s.bottom_margin = Cm(2)
TINTA = RGBColor(0x1C, 0x24, 0x33)

def runs(p, text):
    """Agrega texto con **negrita**, *cursiva* y `código`."""
    for tok in re.split(r'(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+`)', text):
        if not tok: continue
        if tok.startswith('**'): r = p.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith('`'): r = p.add_run(tok[1:-1]); r.font.name = 'Consolas'; r.font.size = Pt(10)
        elif tok.startswith('*'): r = p.add_run(tok[1:-1]); r.italic = True
        else: p.add_run(tok)

def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement('w:shd')
    sh.set(qn('w:val'), 'clear'); sh.set(qn('w:color'), 'auto'); sh.set(qn('w:fill'), hexcolor); tcPr.append(sh)

def tabla(lineas):
    filas = [[c.strip() for c in l.strip().strip('|').split('|')] for l in lineas if not re.match(r'^\|\s*-', l)]
    t = doc.add_table(rows=len(filas), cols=len(filas[0])); t.style = 'Table Grid'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, f in enumerate(filas):
        for j, v in enumerate(f):
            c = t.cell(i, j); c.text = ''; p = c.paragraphs[0]; runs(p, v)
            for r in p.runs: r.font.size = Pt(9.5)
            if i == 0:
                shade(c, '1C2433')
                for r in p.runs: r.bold = True; r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    doc.add_paragraph()

def md(texto):
    lineas = texto.strip().split('\n'); i = 0
    while i < len(lineas):
        l = lineas[i]
        if l.startswith('|'):
            bloque = []
            while i < len(lineas) and lineas[i].startswith('|'): bloque.append(lineas[i]); i += 1
            tabla(bloque); continue
        if l.startswith('#'):
            nivel = len(l) - len(l.lstrip('#'))
            h = doc.add_heading(l.lstrip('#').strip(), level=min(nivel, 3))
            for r in h.runs: r.font.color.rgb = TINTA
        elif re.match(r'^\s*- ', l):
            p = doc.add_paragraph(style='List Bullet'); runs(p, l.strip()[2:])
        elif re.match(r'^\d+\. ', l):
            p = doc.add_paragraph(); p.paragraph_format.left_indent = Cm(0.6); p.paragraph_format.first_line_indent = Cm(-0.6); runs(p, l)
        elif l.strip():
            runs(doc.add_paragraph(), l)
        i += 1

def figura(ruta, pie):
    doc.add_picture(ruta, width=Cm(16.5)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(); r = p.add_run(pie); r.italic = True; r.font.size = Pt(9); p.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Portada
for _ in range(6): doc.add_paragraph()
for txt, tam, neg in [('Combinando Modelos: Hard vs Soft Voting', 24, True), ('Mini informe · Yomi Level: dificultad del japonés en el contenido audiovisual', 13, False),
                      ('', 11, False), ('José Santiago González · Oscar Mateo Arrubla · Leonardo Diasgranados', 12, True),
                      ('Ciencias de la Computación e Inteligencia Artificial · Universidad Sergio Arboleda', 11, False),
                      ('Aprendizaje Supervisado · Docente: Hernán Cruz Bueno', 11, False)]:
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(txt); r.font.size = Pt(tam); r.bold = neg; r.font.color.rgb = TINTA
doc.add_page_break()

md(T.FASE1.replace('## Fase 1 · Exploración del dataset', '## 1. Descripción del dataset'))
figura('figuras/1_dataset.png', 'Figura 1. Distribución de la dificultad con los cortes de los terciles y balance de clases.')
doc.add_page_break()
md(T.FASE2.replace('## Fase 2 · Modelos base', '## 2. Resultados de los modelos base'))
figura('figuras/2_cm_base.png', 'Figura 2. Matrices de confusión de los modelos base (1.807 títulos de prueba).')
doc.add_page_break()
md(T.FASE3.replace('## Fase 3 · Voting Ensemble', '## 3. Resultados de Hard y Soft Voting'))
figura('figuras/3_comparacion.png', 'Figura 3. Accuracy, F1-macro y errores graves de los modelos base y los ensembles.')
figura('figuras/4_cm_voting.png', 'Figura 4. Matrices de confusión de Hard y Soft Voting.')
figura('figuras/5_ablacion.png', 'Figura 5. F1-macro al quitar o agregar modelos del ensemble.')
md(T.DESBALANCE_INTRO)
figura('figuras/6_desbalanceado.png', 'Figura 6. Problema binario "¿es Difícil?": umbral de Soft Voting y falsos positivos/negativos.')
doc.add_page_break()
md(T.FASE5.replace('## Fase 3 (preguntas) y Fase 5 · Análisis y reflexión', '## 4. Análisis y reflexión'))
doc.add_page_break()
md(T.CONCLUSION.replace('## Conclusión', '## 5. Conclusión'))

doc.save('Informe_Taller_Voting_Yomi_Level.docx'); print('docx listo')
