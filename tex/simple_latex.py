import os

def find_boundaries(filename='paper'):
    f = open(filename+'.tex')
    l = f.readline().replace('\r','')
    i, i_limit = 0, 10000

    while (l!='\\section*{Abstract}\n') and (l!='\\section*{Key points summary}\n') and (i<i_limit):
        l = f.readline().replace('\r','')
        i+=1
    MANUSCRIPT_START = i
    while (l!='\\section*{Introduction}\n') and (i<i_limit):
        print l
        l = f.readline().replace('\r','')
        i+=1
    INTRODUCTION_START = i+1
    while (l!='\\section*{References}\n') and (l!='\\begin{thebibliography}{}\n') and (i<i_limit):
        l = f.readline().replace('\r','')
        i+=1
    MANUSCRIPT_END = i
    return MANUSCRIPT_START, INTRODUCTION_START, MANUSCRIPT_END

def reformat_line(args, line):
    if len(line.split('\\label{sec-'))>1:
        return ''
    if (len(line.split('\\includegraphics'))>1) and not args.graphics:
        return ''
    # if (len(line.split('\\href'))>1) and not args.hyperref:
    #     return line.replace('href', 'texttt')
    # if (len(line.split('\\url'))>1) and not args.hyperref:
    #     return line.replace('url', 'texttt')
    else:
        return line

def produce_tex_file(args, replace=True,\
                     with_biblio=None, full_file=False):
    MANUSCRIPT_START, INTRODUCTION_START, MANUSCRIPT_END = \
      find_boundaries(filename=args.filename)
    f = open(args.filename+'.tex')
    # empty read
    for i in range(MANUSCRIPT_START):
        f.readline().replace('\r','')
    # manuscript read
    core_manuscript = ''

    if full_file:
        if args.hyperref:
            core_manuscript += '\\documentclass[a4paper, colorlinks]{article} \n'
            core_manuscript += '\\usepackage{hyperref} \n'
            core_manuscript += '\\hypersetup{allcolors = blue}'
        elif args.plos:
            p = open('plos_template.tex')
            # empty read
            for i in range(200):
                core_manuscript += p.readline()
        else:
            core_manuscript += '\\documentclass[a4paper]{article} \n'

        if args.bibstyle=='apalike' and not args.plos:
            core_manuscript += '\\bibliographystyle{apalike}\n'
        elif args.bibstyle=='biology':
            core_manuscript += '\\usepackage{biology_citations}\n'
            core_manuscript += '\\bibliographystyle{biology_citations}\n'
        if args.lineno:
            core_manuscript += '\\usepackage{lineno} \n'
        if args.graphics:
            core_manuscript += '\\usepackage[demo]{graphicx} \n'
        if not args.no_ams and not args.plos:
            core_manuscript += '\\usepackage{amsmath,amssymb} \n'
        if not args.plos:
            core_manuscript += '\\usepackage[utf8]{inputenc} \n'
            core_manuscript += '\\begin{document} \n'

    for i in range(INTRODUCTION_START-MANUSCRIPT_START):
        core_manuscript += reformat_line(args, f.readline().replace('\r',''))
    if args.lineno:
        core_manuscript += '\\linenumbers \n'
    for i in range(MANUSCRIPT_END-INTRODUCTION_START):
        core_manuscript += reformat_line(args, f.readline().replace('\r',''))

    if args.lineno or args.plos:
        core_manuscript += '\\nolinenumbers \n'

    if with_biblio is not None:
        core_manuscript += open(with_biblio).read()
    else:
        core_manuscript += '\\bibliography{biblio}\n'
    
    if full_file:
        core_manuscript += '\\end{document} \n'
    
    # then some replacements
    if replace:
        core_manuscript = core_manuscript.replace('./figures/', '../figures/')
        # core_manuscript = core_manuscript.replace('.png', '.eps')
        core_manuscript = core_manuscript.replace('\\citetext', '\\cite')
        core_manuscript = core_manuscript.replace('\\textcolor{red}', '\\textbf')

    new_paper = open('simple_'+args.filename+'.tex', 'w')
    new_paper.write(core_manuscript)
    # new_paper.write("\n \nolinenumbers")
    new_paper.close()

def run_compilation(filename='paper'):
    
    
    os.system('pdflatex -shell-escape -interaction=nonstopmode '+filename+'.tex')
    os.system('bibtex -terse '+filename+'.aux')
    os.system('pdflatex -shell-escape -interaction=nonstopmode '+filename+'.tex')
    os.system('pdflatex -shell-escape -interaction=nonstopmode '+filename+'.tex')
    
    return None

if __name__=='__main__':

    import argparse
    # First a nice documentation 
    parser=argparse.ArgumentParser(description=
     """ 
     Generating random sample of a given distributions and
     comparing it with its theoretical value
     """
    ,formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--filename", help=" filename to simplify ", default='paper')
    parser.add_argument("--hyperref", help="with hyperref ?", action="store_true")
    parser.add_argument("--lineno", help="with line numbering ?", action="store_true")
    parser.add_argument("--plos", help="plos submission ?", action="store_true")
    parser.add_argument("--graphics", help="with fake demo graphics ?", action="store_true")
    parser.add_argument("--no_ams", help="remove amsmath", action="store_true")
    parser.add_argument("--no_biblio", help="remove bibliography", action="store_false")
    parser.add_argument("--bibstyle",\
                        help="bibliography style: apalike or biology_citations ",\
                        default='apalike')
    
    
    args = parser.parse_args()
    
    produce_tex_file(args, full_file=True, replace=True)
    os.system('pdflatex -shell-escape -interaction=nonstopmode simple_paper.tex')
    os.system('bibtex -terse simple_paper.aux')
    
    produce_tex_file(args,\
                     with_biblio='simple_paper.bbl', full_file=True, replace=True)
    run_compilation(filename='simple_paper')
