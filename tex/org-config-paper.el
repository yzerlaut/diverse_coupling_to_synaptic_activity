(require 'org)
(defcustom org-latex-default-figure-position "tb!"           "Default position for latex figures."        :group 'org-export-latex                   :type 'string)
(setq org-export-latex-hyperref-format "\\ref{%s}")

;; tools for CITATIONS
(org-add-link-type "cite"   (defun follow-cite (name)     "Open bibliography and jump to appropriate entry.        The document must contain ibliography{filename} somewhere        for this to work"       (find-file-other-window        (save-excursion          (beginning-of-buffer)          (save-match-data            (re-search-forward "\\\\bibliography{\\([^}]+\\)}")            (concat (match-string 1) ".bib"))))       (beginning-of-buffer)       (search-forward name))     (defun export-cite (path desc format)       "Export [[cite:cohen93]] as \cite{cohen93} in LaTeX."       (if (eq format 'latex)           (if (or (not desc) (equal 0 (search "cite:" desc)))               (format "\\cite{%s}" path)             (format "\\cite[%s]{%s}" desc path)))))
(org-add-link-type "citetext"   (defun follow-citetext (name)     "Open bibliography and jump to appropriate entry.        The document must contain ibliography{filename} somewhere        for this to work"       (find-file-other-window        (save-excursion          (beginning-of-buffer)          (save-match-data            (re-search-forward "\\\\bibliography{\\([^}]+\\)}")            (concat (match-string 1) ".bib"))))       (beginning-of-buffer)       (search-forward name))     (defun export-citetext (path desc format)       "Export [[citetext:cohen93]] as \citetext{cohen93} in LaTeX."       (if (eq format 'latex)           (if (or (not desc) (equal 0 (search "citetext:" desc)))               (format "\\citetext{%s}" path)             (format "\\citetext[%s]{%s}" desc path)))))
;;(setq org-latex-link-with-unknown-path-format "\\colorbox{green}{%s}")
(setq org-latex-link-with-unknown-path-format "\\textcolor{red}{\\emph{%s}}")

;; tools for CITATIONS
(defcustom org-beamer-environments-extra nil
  "Environments triggered by tags in Beamer export.
Each entry has 4 elements:

name    Name of the environment
key     Selection key for `org-beamer-select-environment'
open    The opening template for the environment, with the following escapes
        %a   the action/overlay specification
        %A   the default action/overlay specification
        %R   the raw BEAMER_act value
        %o   the options argument, with square brackets
        %O   the raw BEAMER_opt value
        %h   the headline text
        %r   the raw headline text (i.e. without any processing)
        %H   if there is headline text, that raw text in {} braces
        %U   if there is headline text, that raw text in [] brackets
close   The closing string of the environment."
  :group 'org-export-beamer
  :version "24.4"
  :package-version '(Org . "8.1")
  :type '(repeat
	  (list
	   (string :tag "Environment")
	   (string :tag "Selection key")
	   (string :tag "Begin")
	   (string :tag "End"))))
(add-to-list 'org-beamer-environments-extra
               '("onlyenv" "O" "\\begin{onlyenv}%a" "\\end{onlyenv}"))

;; (org-add-link-type \"ref\"
;;       (defun export-cite (path desc format)
;;         \"Export [[ref:intro]] as \ref{intro} in LaTeX.\"
;;         (if (eq format 'latex)
;;             (if (or (not desc) (equal 0 (search \"ref:\" desc)))
;;                 (format \"\\\\ref{%s}\" path)
;;               (format \"\\\\ref[%s]{%s}\" desc path)))))
;; (org-add-link-type \"label\"
;;       (defun export-cite (path desc format)
;;         \"Export [[label:intro]] as \label{intro} in LaTeX.\"
;;         (if (eq format 'latex)
;;             (if (or (not desc) (equal 0 (search \"label:\" desc)))
;;                 (format \"\\\\label{%s}\" path)
;;               (format \"\\\\label[%s]{%s}\" desc path)))))
