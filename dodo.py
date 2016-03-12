from doit.action import CmdAction
import os

# SCRIPTS = ['paper', 'highlights', 'supplementary', 'report', 'litterature']
# SCRIPTS = ['paper', 'presentation', 'highlights']
SCRIPTS = ['paper', 'supplementary']

EXT = 'png' # extension for figure export : png or pdf !
DPI = 300 # resolution for bitmap figures

SVG_FILES =[]
for f in os.listdir('./figures/'):
    if f.endswith('.svg'):
        SVG_FILES.append('figures/'+f)
FILES = [f.replace('.svg', '.'+EXT) for f in SVG_FILES]

#############################################
##### creating the needed arborescence
#############################################

def task_create_necessary_folders():
    DIRS = ['tex/', 'pdf_output/']
    for directory in DIRS:
        if not os.path.exists(directory):
            os.makedirs(directory)

#############################################
##### exporting the figures to a widespread format
#############################################

CMD = 'inkscape  --export-area-drawing '
if EXT=='pdf':
    CMD +='--export-pdf='
elif EXT=='png':
    CMD += ' --export-dpi '+str(DPI)+' --export-png='

def Build_task_for_fig_generation(SVG_FILE, FILE):
    return {'file_dep': [SVG_FILE], 'actions': [CMD+FILE+' '+SVG_FILE], 'targets':[FILE]}

def gen_all_svg_tasks():
    for SVG_FILE, FILE in zip(SVG_FILES, FILES):
        T = Build_task_for_fig_generation(SVG_FILE, FILE)
        T['basename'] = 'SVG figure export --- '+SVG_FILE
        yield T
        
def task_all_svg_png():
    yield gen_all_svg_tasks()
    
#############################################
##### compilation of the org-mode code
#############################################

def build_task_to_generate_tex(filename):
    DEPS = SVG_FILES
    DEPS = DEPS+[filename+".org", 'tex/biblio.bib']
    if filename=='presentation':
        org_cmd0 = 'emacs --batch -l tex/org-config-'+filename+'.el'
        org_cmd = 'org-beamer-export-to-latex'
    else:
        org_cmd0 = 'emacs --batch -l tex/org-config-'+filename+'.el'
        org_cmd = 'org-latex-export-to-latex'
    return {'actions': [CmdAction("cp "+filename+".org tex/"+filename+".org"),
                        CmdAction(org_cmd0+" --file tex/"+filename+".org -f "+org_cmd)],\
            'file_dep': DEPS,
            'targets': ["tex/"+filename+".tex"]}

def Build_task_for_pdflatex_compilation(filename):
    def func0():
        os.system("echo '\RequirePackage{lineno}' > tex/preamble")
        os.system("cp tex/"+filename+".tex tex/"+filename+"2.tex")
        os.system("cat tex/preamble tex/"+filename+"2.tex > tex/"+filename+".tex")
        return True
    def func1():
        os.system("pdflatex -shell-escape -interaction=nonstopmode -output-directory=tex tex/"+filename+".tex > tex/compil_output")
        return True
    def func2():
        os.system("bibtex -terse tex/"+filename+".aux")
        return None
    def func3():
        os.system("mv tex/"+filename+".pdf pdf_output/"+filename+".pdf")
        return None
    return {'actions': [func0, func1, func2, func1, func1, func3],
            'file_dep': SVG_FILES+[filename+'.org'],
            'targets':['pdf_output/'+filename+'.pdf'],
            "clean": True,
            # force doit to always mark the task
            # as up-to-date (unless target removed)
            'uptodate': [True]}

def gen_all_org_to_tex_tasks():
    for script in SCRIPTS:
        T = build_task_to_generate_tex(script)
        T['basename'] = 'from Org-Mode to TeX --- '+script
        yield T
        
def task_all_org_to_tex():
    yield gen_all_org_to_tex_tasks()

def gen_all_tex_to_pdf_tasks():
    for script in SCRIPTS:
        T = Build_task_for_pdflatex_compilation(script)
        T['basename'] = 'from TeX to Pdf --- '+script
        yield T
        
def task_all_tex_to_pdf():
    yield gen_all_tex_to_pdf_tasks()

