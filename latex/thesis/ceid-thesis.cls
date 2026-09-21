\NeedsTeXFormat{LaTeX2e}[1996/12/01]
\ProvidesClass{ceid-thesis}[2026/09/21 CEID thesis class]

\RequirePackage{graphicx}
\RequirePackage{amsmath}
\RequirePackage{fancyhdr}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions
\LoadClass[a4paper,11pt]{book}

% Fonts and languages (Greek/English bilingual document via polyglossia)
\RequirePackage{fontspec}
\RequirePackage{polyglossia}
\setmainlanguage[variant=monotonic]{greek}
\setotherlanguage{english}
\newcommand{\en}[1]{\foreignlanguage{english}{#1}}

\setmainfont{CMU Serif}
\setsansfont{CMU Sans Serif}
\setmonofont{CMU Typewriter Text}

\pagestyle{fancy}

% Title page and front matter fields, set via \supervisor{...} etc. in thesis.tex
\newcommand{\thesis@newfield}[1]{%
    \expandafter\gdef\csname @#1\endcsname{}%
    \expandafter\newcommand\csname #1\endcsname[1]{\expandafter\gdef\csname @#1\endcsname{##1}}%
}
\thesis@newfield{supervisor}
\thesis@newfield{supervisorSecond}
\thesis@newfield{epitropiF}
\thesis@newfield{epitropiS}
\thesis@newfield{subtitle}
\thesis@newfield{translator}
\thesis@newfield{institution}
\thesis@newfield{department}
\thesis@newfield{division}
\thesis@newfield{lab}
\thesis@newfield{other}
\thesis@newfield{uppertitleback}
\thesis@newfield{middletitleback}
\thesis@newfield{lowertitleback}
\thesis@newfield{approval}

\newcommand{\new@tpage}{\newpage\thispagestyle{empty}\null}

% Symmetric 30mm margins on all sides
\RequirePackage[
    a4paper,
    margin=30mm,
    headheight=14pt,
    headsep=25pt,
    footskip=30pt,
]{geometry}

% Title page (\maketitle): cover, back-of-cover, approval and signature pages
\newcommand{\thesis@titleblock}{
    \begin{center}
        \LARGE \bfseries \@title
    \end{center}
    \ifx\@subtitle\@empty\else
    \begin{center}
        \LARGE \@subtitle
    \end{center}
    \fi
    \vspace{5mm}
    \begin{center}
        \huge{\textsc{\@translator}}\\
        \vspace{3mm}
        \normalsize{\toutis}\\
        \vspace{3mm}
        \large{\textbf{\authorNameCapital}}
    \end{center}
    \vspace{5mm}
    \hspace{1mm}
}
\newcommand{\thesis@supervisorblock}{
    \begin{minipage}{15cm}
    \begin{tabbing}
        \large \textbf{\supervisorname :}
        \=\@supervisor\\
        \>\@supervisorSecond\\
    \end{tabbing}
    \end{minipage}
    \vspace{1mm}
}
\renewcommand{\maketitle}{
    \begin{titlepage}
        \let\footnotesize\small
        \let\footnoterule\relax
        \renewcommand{\thefootnote}{\fnsymbol{footnote}}
        \new@tpage
        \begin{center}

        \begin{minipage}{2.5cm}
            \includegraphics[width=2.5cm]{static/figures/PatrasLogo.png}
			\vspace{5pt}
        \end{minipage}

        \begin{minipage}{\dimexpr\textwidth-2.5cm\relax}
        \begin{center}
            \textsc{
            \Large{\@institution} \\
            \large{\@department \\
            \@division
            }}
        \end{center}
        \end{minipage}

        \vfill

                \thesis@titleblock
           \end{center}

           \vfill

                \thesis@supervisorblock

            \vfill

            \begin{center}
                \textsc{\@lab}\\
                \thesisPlaceDate
            \end{center}
            \cleardoublepage

            \new@tpage
                \noindent
                \begin{minipage}[t]{\textwidth}
                    \@uppertitleback
                \end{minipage}
                \vfill
                \thesis@titleblock

           \vfill

                \thesis@supervisorblock
           \vfill

                \begin{minipage}{\textwidth}
                    \@approval
                \end{minipage}

            \vfill
            \begin{center}
                \thesisPlaceDate
            \end{center}

            \cleardoublepage
            \new@tpage
                \noindent
                \begin{minipage}[t]{\textwidth}
                    \@uppertitleback
                \end{minipage}
                \vfill
                \noindent
                \begin{minipage}{\textwidth}
                    \@middletitleback
                \end{minipage}
                \vfill
                \noindent
                \begin{minipage}[b]{\textwidth}
                    \@lowertitleback
                \end{minipage}
    \end{titlepage}
    \renewcommand{\thefootnote}{\arabic{footnote}}
    \setcounter{footnote}{0}
    \gdef\@thanks{}\gdef\@author{}\gdef\@translator{}\gdef\@institution{}
    \gdef\@department{}\gdef\@division{}\gdef\@lab{}\gdef\@other{}
  \gdef\@uppertitleback{}\gdef\@lowertitleback{}%
  \gdef\@title{}\gdef\@subtitle{}\let\maketitle\relax%
}

% Localized names (Greek labels, with English counterparts for \en{} spans)
\newcommand{\keywordsname}{Λέξεις Κλειδιά}
\newcommand{\keywordsnameeng}{\en{Keywords}}
\newcommand{\abstractname}{Περίληψη}
\newcommand{\abstractnameeng}{\en{Abstract}}
\newcommand{\acknowledgementsname}{Ευχαριστίες}
\newcommand{\supervisorname}{\supervisorMaleFemale}
\translator{Διπλωματική Εργασία}
\institution{ΠΑΝΕΠΙΣΤΗΜΙΟ ΠΑΤΡΩΝ}
\department{ΠΟΛΥΤΕΧΝΙΚΗ ΣΧΟΛΗ}
\division{ΤΜΗΜΑ ΜΗΧΑΝΙΚΩΝ ΗΛΕΚΤΡΟΝΙΚΩΝ ΥΠΟΛΟΓΙΣΤΩΝ ΚΑΙ ΠΛΗΡΟΦΟΡΙΚΗΣ}

% Default content for the title-back pages
\uppertitleback{
    \begin{minipage}{2cm}
        \includegraphics[width=2cm]{static/figures/PatrasLogo.png}
    \end{minipage}
    \hspace{0.5cm}
    \begin{minipage}{12cm}
        \@institution \\
        \@department \\
        \@division
    \end{minipage}
}

\approval{
    Εγκρίθηκε από την τριμελή εξεταστική επιτροπή τον \examinationDate.
    % \par: tabular* doesn't end the paragraph on its own
    \par
    \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ccc}
        \textit{(Υπογραφή)} & \textit{(Υπογραφή)} & \textit{(Υπογραφή)} \\[3em]
        ............................. & ............................. & ............................. \\
        \@supervisor & \@epitropiF & \@epitropiS \\
        \supervisorTitle & \epitropiFTitle & \epitropiSTitle \\[4em]
    \end{tabular*}
}

\middletitleback{
    \vspace{3cm}
    \en{Copyright} \en{\copyright}\en{--All rights reserved}\ \ \@author, \copyrightYear.\\
    Με την επιφύλαξη παντός δικαιώματος.\\[12pt]
    Απαγορεύεται η αντιγραφή, αποθήκευση και διανομή της παρούσας εργασίας, εξ
    ολοκλήρου ή τμήματος αυτής, για εμπορικό σκοπό.  Επιτρέπεται η ανατύπωση,
    αποθήκευση και διανομή για σκοπό μη κερδοσκοπικό, εκπαιδευτικής ή
    ερευνητικής φύσης, υπό την προϋπόθεση να αναφέρεται η πηγή προέλευσης και να
    διατηρείται το παρόν μήνυμα.
    \\[24pt]
    Το περιεχόμενο αυτής της εργασίας δεν απηχεί απαραίτητα τις απόψεις του Τμήματος, του Επιβλέποντα, ή της επιτροπής που την ενέκρινε.
    \\[24pt]
    \textbf{Υπεύθυνη Δήλωση}
    \\
Βεβαιώνω ότι είμαι συγγραφέας αυτής της πτυχιακής εργασίας, και ότι κάθε
βοήθεια την οποία είχα για την προετοιμασία της είναι πλήρως αναγνωρισμένη και αναφέρεται
στην πτυχιακή εργασία. Επίσης έχω αναφέρει τις όποιες πηγές από τις οποίες έκανα χρήση
δεδομένων, ιδεών ή λέξεων, είτε αυτές αναφέρονται ακριβώς είτε παραφρασμένες. Επίσης,
βεβαιώνω ότι αυτή η πτυχιακή εργασία προετοιμάστηκε από εμένα προσωπικά ειδικά για τις
απαιτήσεις του προγράμματος σπουδών του τμήματος Μηχανικών Ηλεκτρονικών Υπολογιστών και Πληροφορικής του Πανεπιστημίου Πατρών.
    \\[24pt]
    \textit{(Υπογραφή)}
    \\[24pt]
    .............................\\
    \@author
}

% Front/main/back matter redefinitions (roman vs. arabic page numbering)
\renewcommand{\frontmatter}{
    \cleardoublepage
    \@mainmatterfalse \pagenumbering{roman}
}
\renewcommand{\mainmatter}{
    \label{lastfront}
    \cleardoublepage
    \@mainmattertrue\pagenumbering{arabic}
    \addtocontents{toc}{\protect\addvspace{1em}}
}
\renewcommand{\backmatter}{
    \if@openright\cleardoublepage\else\clearpage\fi
    \@mainmatterfalse
}

% Chapter start pages, running headers/footers (fancyhdr)
% emptypage: no-op under oneside, only matters if twoside is re-enabled
\RequirePackage{emptypage}
\renewcommand{\chaptermark}[1]{
    \markboth{\mbox{\selectlanguage{greek}\@chapapp}\ \thechapter.\ \ #1}{}
}
\renewcommand{\sectionmark}[1]{
    \markright{\thesection\ \ #1}{}
}

% Chapter name left, page number right
\fancyhf{}
\fancyhead[L]{\slshape\nouppercase{\leftmark}}
\fancyhead[R]{\thepage}
\AtBeginDocument{\setlength{\headwidth}{\textwidth}}

% Table of contents / list of figures / list of tables (add themselves to TOC)
\AddToHook{cmd/tableofcontents/after}{\addcontentsline{toc}{chapter}{\contentsname}}
\AddToHook{cmd/listoffigures/after}{\addcontentsline{toc}{chapter}{\listfigurename}}
\AddToHook{cmd/listoftables/after}{\addcontentsline{toc}{chapter}{\listtablename}}

% Front-matter environments: keywords, acknowledgements, abstract(s)
\newenvironment{keywords}{\section*{\keywordsname}}{}
\newenvironment{keywordseng}{\section*{\keywordsnameeng}}{}
\newcommand{\footacknowledgement}[1]{%
  \let\@tmp@makefntext\@makefntext%
  \long\def\@makefntext##1{%
      \parindent 1em%
      \noindent##1}%
  \footnotetext{#1}%
  \let\@makefntext\@tmp@makefntext%
}
\newenvironment{acknowledgements}{%
  \chapter*{\acknowledgementsname}%
  \addcontentsline{toc}{chapter}{\acknowledgementsname}%
}{}
\newenvironment{abstract}{%
  \chapter*{\abstractname\@mkboth{\abstractname}{\abstractname}}
  \addcontentsline{toc}{chapter}{\abstractname}
}{\par\vfil\null\endtitlepage}
% No trailing page break here - it would leave a bare page
\newenvironment{abstracteng}{%
  \chapter*{\abstractnameeng\@mkboth{\abstractnameeng}{\abstractnameeng}}
  \addcontentsline{toc}{chapter}{\abstractnameeng}
}{}

% Dummy label for \pageref{lastback}, mirrors \label{lastfront}
\AtEndDocument{\phantomsection\label{lastback}}

% Dedication page
\newcommand{\thesisDedication}[1]{%
	\cleardoublepage
	\thispagestyle{empty}
    \vspace*{\fill}
    \begin{flushright}
    	\textit{#1}
    \end{flushright}
    \vspace*{\fill}
    \clearpage
}

\endinput
