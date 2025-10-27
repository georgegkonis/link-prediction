%%%%%%
% Template for typesetting Diploma Theses
%
% John Liaperdos, October 2014
%
% (adapted from the NTUA template for diploma theses [http://web.dbnet.ntua.gr/el/diplomas.html])
%
% Last update: Oct 25, 2014
%
%%%%%%
% Class Information
\def\filename{ceid-thesis}
\def\fileversion{1.0}
\def\filedate{2014/10/25}
\NeedsTeXFormat{LaTeX2e}[1996/12/01]
\ProvidesClass{ceid-thesis}[%
    \filedate\space\fileversion\space LaTeX document class.]
% Required Packages
\RequirePackage{graphicx}
\RequirePackage{amsmath}
\RequirePackage{fancyhdr}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions
\LoadClass[a4paper,11pt]{book}


\usepackage{fontspec}
\usepackage{polyglossia}
\setmainlanguage[variant=monotonic]{greek}
\setotherlanguage{english}

\setmainfont{CMU Serif}
\setsansfont{CMU Sans Serif}
\setmonofont{CMU Typewriter Text}


\pagestyle{fancy}
% Title Page and Front Matter
\def\@supervisor{}              \def\supervisor#1{\gdef\@supervisor{#1}}
\def\@supervisorSecond{}        \def\supervisorSecond#1{\gdef\@supervisorSecond{#1}}

\def\@epitropiF{}               \def\epitropiF#1{\gdef\@epitropiF{#1}}
\def\@epitropiS{}               \def\epitropiS#1{\gdef\@epitropiS{#1}}
\def\@subtitle{}                    \def\subtitle#1{\gdef\@subtitle{#1}}
\def\@translator{}              \def\translator#1{\gdef\@translator{#1}}
\def\@institution{}             \def\institution#1{\gdef\@institution{#1}}
\def\@department{}              \def\department#1{\gdef\@department{#1}}
\def\@division{}                    \def\division#1{\gdef\@division{#1}}
\def\@lab{}                             \def\lab#1{\gdef\@lab{#1}}
\def\@other{}                           \def\other#1{\gdef\@other{#1}}
\def\@uppertitleback{}      \long\def\uppertitleback#1{\def\@uppertitleback{#1}}
\def\@middletitleback{}   \long\def\middletitleback#1{\def\@middletitleback{#1}}
\def\@lowertitleback{}    \long\def\lowertitleback#1{\def\@lowertitleback{#1}}
\def\@approval{}   \long\def\approval#1{\def\@approval{#1}}
\def\@mysig{}             \long\def\mysig#1{\def\@mysig{#1}}

\def\new@tpage{\newpage\thispagestyle{empty}\null}
% Measurements
% Equal odd/even margins: same left/right margin on every page (no extra
% binding gutter), unlike a typical twoside book layout.
\setlength{\oddsidemargin}{5mm}
\setlength{\evensidemargin}{5mm}
\setlength{\textwidth}{150mm}

\setlength{\topmargin}{0mm}
\setlength{\textheight}{235mm}
% Header eats into the top margin; shrink topmargin so body text still
% starts at the intended position.
\advance \topmargin by -\headheight
\advance \topmargin by -\headsep

% Maketitle
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

        \begin{minipage}{15cm}
        \begin{center}
            \textsc{
            \Large{\@institution} \\
            \large{\@department \\
            \@division \\
            %\@other
            }}
        \end{center}
        \end{minipage}
        %\hfill %\hspace{0.5cm}

        %
        \vfill
        %
                %
                \begin{center}
                    \LARGE \bfseries \@title
                \end{center}
                \ifx\@subtitle\@empty\else
                \begin{center}
                    \LARGE \@subtitle
                \end{center}
                \fi
                %
                \vspace{5mm}
                %
                \begin{center}
                    \huge{\textsc{\@translator}}\\
                    \vspace{3mm}
                    \normalsize{\toutis}\\
                    \vspace{3mm}
                    \large{\textbf{\authorNameCapital}}
                \end{center}
                %
                \vspace{5mm}
                %
                \hspace{1mm}
           \end{center}

           \vfill

                \begin{minipage}{15cm}
                \begin{tabbing}
                    \large \textbf{\supervisorname :} 
                    \=\@supervisor\\
                    \>\@supervisorSecond\\
                \end{tabbing}
                \end{minipage}
                \vspace{1mm}
   
            %
            \vfill
            %

            \begin{center}
                \textsc{\@lab}\\
                \thesisPlaceDate
            \end{center}
            %
            \cleardoublepage

            \if@twoside\new@tpage
                \noindent
                \begin{minipage}[t]{\textwidth}
                    \@uppertitleback
                \end{minipage}
                \par
                \vfill
               \begin{center}
                    \LARGE \bfseries \@title
                \end{center}
                \ifx\@subtitle\@empty\else
                \begin{center}
                    \LARGE \@subtitle
                \end{center}
                \fi
                %
                \vspace{5mm}
                %
                \begin{center}
                    \huge{\textsc{\@translator}}\\
                    \vspace{3mm}
                    \normalsize{\toutis}\\
                    \vspace{3mm}
                    \large{\textbf{\authorNameCapital}}
                \end{center}
                %
                \vspace{5mm}
                %
                \hspace{1mm}

           \vfill

                \begin{minipage}{15cm}
                \begin{tabbing}
                    \large \textbf{\supervisorname :} 
                    \=\@supervisor\\
                    \>\@supervisorSecond\\
                \end{tabbing}
                \end{minipage}
                \vspace{1mm}
           \vfill

                \begin{minipage}{\textwidth}
                    \@approval
                \end{minipage}

            \vfill
            \begin{center}
                \thesisPlaceDate
            \end{center}
            %
            \fi
           \new@tpage
           \begin{minipage}[b]{\textwidth}
             \@mysig
           \end{minipage}

            \cleardoublepage
            \if@twoside\new@tpage
                \noindent
                \begin{minipage}[t]{\textwidth}
                    \@uppertitleback
                \end{minipage}
                \par
                \vfill
                \noindent
                \begin{minipage}{\textwidth}
                    \@middletitleback
                \end{minipage}
                \par
                \vfill
                \noindent
                \begin{minipage}[b]{\textwidth}
                    \@lowertitleback
                \end{minipage}
            \fi
    \end{titlepage}
    %
    \renewcommand{\thefootnote}{\arabic{footnote}}
    \setcounter{footnote}{0}
    %
    \gdef\@thanks{}\gdef\@author{}\gdef\@translator{}\gdef\@institution{}
    \gdef\@department{}\gdef\@division{}\gdef\@lab{}\gdef\@other{}
  \gdef\@uppertitleback{}\gdef\@lowertitleback{}%
  \gdef\@title{}\gdef\@subtitle{}\let\maketitle\relax%
}


% Names
\newcommand{\keywordsname}{Λέξεις Κλειδιά}
\newcommand{\keywordsnameeng}{\en{Keywords}}
\newcommand{\abstractname}{Περίληψη}
\newcommand{\abstractnameeng}{\en{Abstract}}
\newcommand{\acknowledgementsname}{Ευχαριστίες}
\newcommand{\supervisorname}{\supervisorMaleFemale}
% More Names
\translator{Διπλωματική Εργασία}
\institution{ΠΑΝΕΠΗΣΤΗΜΙΟ ΠΑΤΡΩΝ}
\department{ΠΟΛΥΤΕΧΝΙΚΗ ΣΧΟΛΗ}
\division{ΤΜΗΜΑ ΜΗΧΑΝΙΚΩΝ ΗΛΕΚΤΡΟΝΙΚΩΝ ΥΠΟΛΟΓΙΣΤΩΝ ΚΑΙ ΠΛΗΡΟΦΟΡΙΚΗΣ}


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
    Εγκρίθηκε από την τριμελή εξεταστική επιτροπή την \examinationDate.
    \begin{tabbing}
    aaaaaaaaaaaaaaaaaaaaaaaaaaa\=aaaaaaaaaaaaaaaaaaaaaaaaaaa\=\kill
    \textit{(Υπογραφή)}\>   \textit{(Υπογραφή)}\>  \textit{(Υπογραφή)}\\\\\\
    .............................\>  .............................\>  .............................\\
    \@supervisor\>         \@epitropiF\>    \@epitropiS\\
    \supervisorTitle\>    \epitropiFTitle\>      \epitropiSTitle\\\\\\\\
    \end{tabbing}

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
απαιτήσεις του προγράμματος σπουδών του τμήματος Μηχανικών Ηλεκτρωνικών Υπολογιστών και Πληρογορικής του Πανεπηστημίου Πατρών.
    \\[24pt]
    \textit{(Υπογραφή)}
    \\[24pt]
    .............................\\
    \@author
}


% Redefining Commands
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
% Headings
\def\invisiblethingie{\tiny\ }
% A blank page inserted before a new chapter when the page count is even,
% so chapters always start on the right-hand (odd) page when bound/printed.
\def\myemptypage{
    \mbox{}
    \vspace{\fill}
    \begin{flushright}
        \invisiblethingie
    \end{flushright}
    \vspace{5ex}
    \thispagestyle{empty}
}
%
\renewcommand{\cleardoublepage}{
    \clearpage %
    \if@twoside \ifodd\c@page\else
    \myemptypage
    \newpage
    \if@twocolumn\mbox{}\newpage\fi\fi\fi
}
%
\renewcommand{\chaptermark}[1]{
    \markboth{\mbox{\selectlanguage{greek}\@chapapp}\ \thechapter.\ \ #1}{}
%        \markboth{\mbox{\@chapapp}\ \thechapter.\ \ #1}{}
}
\renewcommand{\sectionmark}[1]{
    \markright{\thesection\ \ #1}{}
}
%
\fancyhf{}
\fancyhead[LE,RO]{\thepage}
\fancyhead[LO]{\slshape\nouppercase{{\rightmark}}}
\fancyhead[RE]{\slshape\nouppercase{{\leftmark}}}
\addtolength{\headheight}{2pt}
\setlength{\headwidth}{\textwidth}
\addtolength{\topmargin}{-2pt}
%
\let\finik@tocont\tableofcontents
\renewcommand{\tableofcontents}{
    \finik@tocont
    \addcontentsline{toc}{chapter}{\contentsname}
}

\let\finik@lof\listoffigures
\renewcommand{\listoffigures}{
    \finik@lof
    \addcontentsline{toc}{chapter}{\listfigurename}
}
\let\finik@lot\listoftables
\renewcommand{\listoftables}{
    \finik@lot
    \addcontentsline{toc}{chapter}{\listtablename}
}
% More Enviroments
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
\def\abstract{%
  \chapter*{\abstractname\@mkboth{\abstractname}{\abstractname}}
  \addcontentsline{toc}{chapter}{\abstractname}
}
\def\abstracteng{
  \chapter*{\abstractnameeng\@mkboth{\abstractnameeng}{\abstractnameeng}}
  \addcontentsline{toc}{chapter}{\abstractnameeng}
}
\def\endabstract{\par\vfil\null\endtitlepage}

% Misc

% Dummy label marking the last page, for anyone who wants \pageref{lastback}
% (mirrors \label{lastfront} set in \mainmatter above).
\AtEndDocument{\subparagraph{}\label{lastback}}

%% Dedication
\newcommand{\thesisDedication}[1]{%
	\cleardoublepage
	\thispagestyle{empty}
    ~\\
    \vfill
    \begin{flushright}
    	\textit{#1}
    \end{flushright}
    ~\\
    \vfill
    \pagebreak
}



\endinput