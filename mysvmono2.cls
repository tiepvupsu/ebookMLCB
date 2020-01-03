% SVMONO DOCUMENT CLASS -- version 4.17 (31-Oct-06)
% Springer Verlag global LaTeX2e support for monographs
%%
%%
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
%%
\usepackage[utf8]{inputenc}
%\usepackage[vietnamese]{babel}
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{svmono}[2006/10/31 v4.17
^^JSpringer Verlag global LaTeX document class for monographs]

% Options
% citations
\DeclareOption{natbib}{\ExecuteOptions{oribibl}%
\AtEndOfClass{% Loading package 'NATBIB'
\RequirePackage{natbib}
% Changing some parameters of NATBIB
\setlength{\bibhang}{\parindent}
%\setlength{\bibsep}{0mm}
\let\bibfont=\small
\def\@biblabel#1{#1.}
\newcommand{\etal}{\textit{et al}.}
%\bibpunct[,]{(}{)}{;}{a}{}{,}}}
}}
% Springer environment
\let\if@spthms\iftrue
\DeclareOption{nospthms}{\let\if@spthms\iffalse}
%
\let\envankh\@empty   % no anchor for "theorems"
%
\let\if@envcntreset\iffalse % environment counter is not reset
\let\if@envcntresetsect=\iffalse % reset each section?
\DeclareOption{envcountresetchap}{\let\if@envcntreset\iftrue}
\DeclareOption{envcountresetsect}{\let\if@envcntreset\iftrue
\let\if@envcntresetsect=\iftrue}
%
\let\if@envcntsame\iffalse  % NOT all environments work like "Theorem",
                            % each using its own counter
\DeclareOption{envcountsame}{\let\if@envcntsame\iftrue}
%
\let\if@envcntshowhiercnt=\iffalse % do not show hierarchy counter at all
%
% enhance theorem counter
\DeclareOption{envcountchap}{\def\envankh{chapter}% show \thechapter along with theorem number
\let\if@envcntshowhiercnt=\iftrue
\ExecuteOptions{envcountreset}}
%
\DeclareOption{envcountsect}{\def\envankh{section}% show \thesection along with theorem number
\let\if@envcntshowhiercnt=\iftrue
\ExecuteOptions{envcountreset}}
%
% languages
\let\switcht@@therlang\relax
\let\svlanginfo\relax
\def\ds@deutsch{\def\switcht@@therlang{\switcht@deutsch}%
\gdef\svlanginfo{\typeout{Man spricht deutsch.}\global\let\svlanginfo\relax}}
\def\ds@francais{\def\switcht@@therlang{\switcht@francais}%
\gdef\svlanginfo{\typeout{On parle francais.}\global\let\svlanginfo\relax}}
%
\AtBeginDocument{\@ifpackageloaded{babel}{%
\@ifundefined{extrasamerican}{}{\addto\extrasamerican{\switcht@albion}}%
\@ifundefined{extrasaustralian}{}{\addto\extrasaustralian{\switcht@albion}}%
\@ifundefined{extrasbritish}{}{\addto\extrasbritish{\switcht@albion}}%
\@ifundefined{extrascanadian}{}{\addto\extrascanadian{\switcht@albion}}%
\@ifundefined{extrasenglish}{}{\addto\extrasenglish{\switcht@albion}}%
\@ifundefined{extrasnewzealand}{}{\addto\extrasnewzealand{\switcht@albion}}%
\@ifundefined{extrasUKenglish}{}{\addto\extrasUKenglish{\switcht@albion}}%
\@ifundefined{extrasUSenglish}{}{\addto\extrasUSenglish{\switcht@albion}}%
\@ifundefined{captionsfrench}{}{\addto\captionsfrench{\switcht@francais}}%
\@ifundefined{extrasgerman}{}{\addto\extrasgerman{\switcht@deutsch}}%
\@ifundefined{extrasngerman}{}{\addto\extrasngerman{\switcht@deutsch}}%
\@ifundefined{vietnamese}{}{\addto\extrasvietnamese{\switcht@vietnamese}}%
}{\switcht@@therlang}%
}
% numbering style of floats, equations
\newif\if@numart   \@numartfalse
\DeclareOption{numart}{\@numarttrue}
\def\set@numbering{\if@numart\else\num@book\fi}
\AtEndOfClass{\set@numbering}
% style for vectors
\DeclareOption{vecphys}{\def\vec@style{phys}}
\DeclareOption{vecarrow}{\def\vec@style{arrow}}
% running heads
\let\if@runhead\iftrue
\DeclareOption{norunningheads}{\let\if@runhead\iffalse}
% referee option
\let\if@referee\iffalse
\def\makereferee{\def\baselinestretch{2}\selectfont
\newbox\refereebox
\setbox\refereebox=\vbox to\z@{\vskip0.5cm%
  \hbox to\textwidth{\normalsize\tt\hrulefill\lower0.5ex
        \hbox{\kern5\p@ referee's copy\kern5\p@}\hrulefill}\vss}%
\def\@oddfoot{\copy\refereebox}\let\@evenfoot=\@oddfoot}
\DeclareOption{referee}{\let\if@referee\iftrue
\AtBeginDocument{\makereferee\small\normalsize}}
% modification of thebibliography
\let\if@openbib\iffalse
\DeclareOption{openbib}{\let\if@openbib\iftrue}
% LaTeX standard, sectionwise references
\DeclareOption{oribibl}{\let\oribibl=Y}
\DeclareOption{sectrefs}{\let\secbibl=Y}
%
% footinfo option (provides an informatory line on every page)
\def\SpringerMacroPackageNameA{svmono.cls}
% \thetime, \thedate and \timstamp are macros to include
% time, date (or both) of the TeX run in the document
% \def\maketimestamp{\count255=\time
% \divide\count255 by 60\relax
% \edef\thetime{\the\count255:}%
% \multiply\count255 by-60\relax
% \advance\count255 by\time
% \edef\thetime{\thetime\ifnum\count255<10 0\fi\the\count255}
% \edef\thedate{\number\day-\ifcase\month\or Jan\or Feb\or Mar\or
%              Apr\or May\or Jun\or Jul\or Aug\or Sep\or Oct\or
%              Nov\or Dec\fi-\number\year}
% \def\timstamp{\hbox to\hsize{\tt\hfil\thedate\hfil\thetime\hfil}}}
% \maketimestamp
%
% \footinfo generates a info footline on every page containing
% pagenumber, jobname, macroname, and timestamp
   %  \DeclareOption{footinfo}{\AtBeginDocument{\maketimestamp
   % \def\ps@empty{\let\@mkboth\@gobbletwo
   % \let\@oddhead\@empty\let\@evenhead\@empty}%
   % \def\@oddfoot{\scriptsize\tt Page:\,\thepage\space\hfil
   %               job:\,\jobname\space\hfil
   %               macro:\,\SpringerMacroPackageNameA\space\hfil
   %               date/time:\,\thedate/\thetime}%
   % \let\@evenfoot=\@oddfoot}}
%
% start new chapter on any page
\newif\if@openright \@openrighttrue
\DeclareOption{openany}{\@openrightfalse}
%
% no size changing allowed
% \DeclareOption{11pt}{\OptionNotUsed}
% \DeclareOption{12pt}{\OptionNotUsed}
% options for the article class
\def\@rticle@options{12pt,twoside}
% fleqn
\DeclareOption{fleqn}{\def\@rticle@options{12pt,twoside,fleqn}%
\AtEndOfClass{\let\leftlegendglue\relax}%
\AtBeginDocument{\mathindent\parindent}}
% hanging sectioning titles
\let\if@sechang\iffalse
\DeclareOption{sechang}{\let\if@sechang\iftrue}
\def\ClassInfoNoLine#1#2{%
   \ClassInfo{#1}{#2\@gobble}%
}
\let\SVMonoOpt\@empty
\DeclareOption*{\InputIfFileExists{sv\CurrentOption.clo}{%
\global\let\SVMonoOpt\CurrentOption}{%
\ClassWarning{Springer-SVMono}{Specified option or subpackage
"\CurrentOption" \MessageBreak not found
passing it to article class \MessageBreak
-}\PassOptionsToClass{\CurrentOption}{article}%
}}
\ProcessOptions\relax
\ifx\SVMonoOpt\@empty\relax
\ClassInfoNoLine{Springer-SVMono}{extra/valid Springer sub-package
\MessageBreak not found in option list - using "global" style}{}
\fi
\LoadClass[\@rticle@options]{article}
\raggedbottom

% various sizes and settings for monographs

\setlength{\textwidth}{28pc}   %  11.8cm
%\setlength{\textheight}{12pt}\multiply\textheight by 45\relax
\setlength{\textheight}{540\p@}
\setlength{\topmargin}{0cm}
\setlength\oddsidemargin   {53\p@}
\setlength\evensidemargin  {53\p@}
\setlength\marginparwidth{90\p@}
\setlength\headsep   {12\p@}

\setlength{\parindent}{15\p@}
\setlength{\parskip}{\z@ \@plus \p@}
\setlength{\hfuzz}{2\p@}
\setlength{\arraycolsep}{1.5\p@}

\frenchspacing

\tolerance=500

\predisplaypenalty=0
\clubpenalty=10000
\widowpenalty=10000

\setlength\footnotesep{7.7\p@}

\newdimen\betweenumberspace          % dimension for space between
\betweenumberspace=5\p@               % number and text of titles
\newdimen\headlineindent             % dimension for space of
\headlineindent=2.5cc                % number and gap of running heads

% fonts, sizes, and the like
\renewcommand\small{%
   \@setfontsize\small\@ixpt{11}%
   \abovedisplayskip 8.5\p@ \@plus3\p@ \@minus4\p@
   \abovedisplayshortskip \z@ \@plus2\p@
   \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \parsep \z@ \@plus\p@ \@minus\p@
               \topsep 6\p@ \@plus2\p@ \@minus4\p@
               \itemsep\z@}%
   \belowdisplayskip \abovedisplayskip
}
%
\let\footnotesize=\small
%
\newenvironment{petit}{\par\addvspace{6\p@}\small}{\par\addvspace{6\p@}}
%

% modification of automatic positioning of floating objects
\setlength\@fptop{\z@ }
\setlength\@fpsep{12\p@ }
\setlength\@fpbot{\z@ \@plus 1fil }
\def\textfraction{.01}
\def\floatpagefraction{.8}
\setlength{\intextsep}{20\p@ \@plus 2\p@ \@minus 2\p@}
\setcounter{topnumber}{4}
\def\topfraction{.9}
\setcounter{bottomnumber}{2}
\def\bottomfraction{.7}
\setcounter{totalnumber}{6}
%
% size and style of headings
\newcommand{\partsize}{\Large}
\newcommand{\partstyle}{\bfseries\boldmath}
\newcommand{\chapsize}{\Large}
\newcommand{\chapstyle}{\bfseries\boldmath}
\newcommand{\chapshooksize}{\small}
\newcommand{\chapshookstyle}{\itshape\unboldmath}
\newcommand{\secsize}{\large}
\newcommand{\secstyle}{\bfseries\boldmath}
\newcommand{\subsecsize}{\normalsize}
\newcommand{\subsecstyle}{\bfseries\boldmath}
%
\def\cleardoublepage{\clearpage\if@twoside \ifodd\c@page\else
    \hbox{}\newpage\if@twocolumn\hbox{}\newpage\fi\fi\fi}

\newcommand{\clearemptydoublepage}{%
        \clearpage{\pagestyle{empty}\cleardoublepage}}
\newcommand{\startnewpage}{\if@openright\clearemptydoublepage\else\clearpage\fi}

% redefinition of \part
\renewcommand\part{\clearemptydoublepage
         \thispagestyle{empty}
         \if@twocolumn
            \onecolumn
            \@tempswatrue
         \else
            \@tempswafalse
         \fi
         \@ifundefined{thispagecropped}{}{\thispagecropped}
         \secdef\@part\@spart}

\def\@part[#1]#2{\ifnum \c@secnumdepth >-2\relax
        \refstepcounter{part}
        \addcontentsline{toc}{part}{\partname\
        \thepart\thechapterend\hspace{\betweenumberspace}%
        #1}\else
        \addcontentsline{toc}{part}{#1}\fi
   \markboth{}{}
   {\raggedleft
    \ifnum \c@secnumdepth >-2\relax
      \normalfont\partstyle\partsize\vrule height 34pt width 0pt depth 0pt%
     \partname\ \thepart\llap{\smash{\lower 5pt\hbox to\textwidth{\hrulefill}}}
    \par
    \vskip 128.3\p@ \fi
    #2\par}\@endpart}
%
% \@endpart finishes the part page
%
\def\@endpart{\vfil
   \if@twoside
       \hbox{}
       \thispagestyle{empty}
   \fi
   \if@tempswa
     \twocolumn
   \fi}
%
\def\@spart#1{{\raggedleft
   \normalfont\partsize\partstyle
   #1\par}\@endpart}
%
\newenvironment{partbacktext}{\def\@endpart{\vfil}}
{\thispagestyle{empty}  \if@tempswa\twocolumn\fi}
%
% (re)define sectioning
\setcounter{secnumdepth}{2}

\def\seccounterend{}
\def\seccountergap{\hskip\betweenumberspace}
\def\@seccntformat#1{\csname the#1\endcsname\seccounterend\seccountergap\ignorespaces}
%
\let\firstmark=\botmark
%
\@ifundefined{thechapterend}{\def\thechapterend{}}{}
%
\if@sechang
   \def\sec@hangfrom#1{\setbox\@tempboxa\hbox{#1}%
         \hangindent\wd\@tempboxa\noindent\box\@tempboxa}
\else
   \def\sec@hangfrom#1{\setbox\@tempboxa\hbox{#1}%
         \hangindent\z@\noindent\box\@tempboxa}
\fi

\def\chap@hangfrom#1{\noindent\vrule height 34pt width 0pt depth 0pt
\rlap{\smash{\lower 5pt\hbox to\textwidth{\hrulefill}}}\hbox{#1}
\vskip10pt}
\def\schap@hangfrom{\chap@hangfrom{}}

\newcounter{chapter}
%
\@addtoreset{section}{chapter}
\@addtoreset{footnote}{chapter}

\newif\if@mainmatter \@mainmattertrue
\newcommand\frontmatter{\startnewpage
            \@mainmatterfalse\pagenumbering{roman}
            \setcounter{page}{1}}
%
\newcommand\mainmatter{\clearemptydoublepage
       \@mainmattertrue\pagenumbering{arabic}}
%
\newcommand\backmatter{\clearemptydoublepage\@mainmatterfalse}

\def\@chapapp{\chaptername}

\newdimen\chapstarthookwidth
\newcommand\chapstarthook[2][0.66\textwidth]{%
\setlength{\chapstarthookwidth}{#1}%
\gdef\chapst@rthook{#2}}

\newcommand{\processchapstarthook}{\@ifundefined{chapst@rthook}{}{%
    \setbox0=\hbox{\vbox{\hyphenpenalty=50
    \begin{flushright}
    \begin{minipage}{\chapstarthookwidth}
       \vrule\@width\z@\@height21\p@\@depth\z@
       \normalfont\chapshooksize\chapshookstyle\chapst@rthook
    \end{minipage}
    \end{flushright}}}%
    \@tempdima=\pagetotal
    \advance\@tempdima by\ht0
    \ifdim\@tempdima<106\p@
       \multiply\@tempdima by-1
       \advance\@tempdima by106\p@
       \vskip\@tempdima
    \fi
    \box0\par
    \global\let\chapst@rthook=\undefined}}

\newcommand\chapter{\startnewpage
                    \@ifundefined{thispagecropped}{}{\thispagecropped}
                    \thispagestyle{fancy}%
                    \global\@topnum\z@
                    \@afterindentfalse
                    \secdef\@chapter\@schapter}

\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
                       \refstepcounter{chapter}%
                       \if@mainmatter
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}{\protect
                                  \numberline{\thechapter\thechapterend}#1}%
                       \else
                         \addcontentsline{toc}{chapter}{#1}%
                       \fi
                    \else
                      \addcontentsline{toc}{chapter}{#1}%
                    \fi
                    \chaptermark{#1}%
                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                    \if@twocolumn
                      \@topnewpage[\@makechapterhead{#2}]%
                    \else
                      \@makechapterhead{#2}%
                      \@afterheading
                    \fi}

\def\@schapter#1{\if@twocolumn
                   \@topnewpage[\@makeschapterhead{#1}]%
                 \else
                   \@makeschapterhead{#1}%
                   \@afterheading
                 \fi}

%%changes position and layout of numbered chapter headings
\def\@makechapterhead#1{{\parindent\z@\raggedright\normalfont
  \hyphenpenalty \@M
  \interlinepenalty\@M
  \chapsize\chapstyle
  \chap@hangfrom{\thechapter\thechapterend\hskip\betweenumberspace}%!!!
  \ignorespaces#1\par\nobreak
  \processchapstarthook
  \ifdim\pagetotal>157\p@
     \vskip 11\p@
  \else
     \@tempdima=168\p@\advance\@tempdima by-\pagetotal
     \vskip\@tempdima
  \fi}}

%%changes position and layout of unnumbered chapter headings
\def\@makeschapterhead#1{{\parindent \z@ \raggedright\normalfont
  \hyphenpenalty \@M
  \interlinepenalty\@M
  \chapsize\chapstyle
  \schap@hangfrom
  \ignorespaces#1\par\nobreak
  \processchapstarthook
  \ifdim\pagetotal>157\p@
     \vskip 11\p@
  \else
     \@tempdima=168\p@\advance\@tempdima by-\pagetotal
     \vskip\@tempdima
  \fi}}

% predefined unnumbered headings
\newcommand{\preface}[1][\prefacename]{\chapter*{#1}\markboth{#1}{#1}}
% same with TOC entry
\newcommand{\Preface}[1][\prefacename]{\chapter*{#1}\markboth{#1}{#1}%
\addcontentsline{toc}{chapter}{#1}}

% measures and setting of sections
\renewcommand\section{\@startsection{section}{1}{\z@}%
                       {-24\p@ \@plus -4\p@ \@minus -4\p@}%
                       {12\p@ \@plus 4\p@ \@minus 4\p@}%
                       {\normalfont\secsize\secstyle
                        \rightskip=\z@ \@plus 8em\pretolerance=10000 }}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                       {-17\p@ \@plus -4\p@ \@minus -4\p@}%
                       {10\p@ \@plus 4\p@ \@minus 4\p@}%
                       {\normalfont\subsecsize\subsecstyle
                        \rightskip=\z@ \@plus 8em\pretolerance=10000 }}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                       {-17\p@ \@plus -4\p@ \@minus -4\p@}%
                       {10\p@ \@plus 4\p@ \@minus 4\p@}%
                       {\normalfont\normalsize\subsecstyle
                        \rightskip=\z@ \@plus 8em\pretolerance=10000 }}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                       {-10\p@ \@plus -4\p@ \@minus -4\p@}%
                       {10\p@ \@plus 4\p@ \@minus 4\p@}%
                       {\normalfont\normalsize\itshape
                        \rightskip=\z@ \@plus 8em\pretolerance=10000 }}
\def\subparagraph{\@startsection{subparagraph}{5}{\z@}%
    {-5.388\p@ \@plus-4\p@ \@minus-4\p@}{-5\p@}{\normalfont\normalsize\itshape}}

% Appendix
\renewcommand\appendix{\par
                \stepcounter{chapter}
                \setcounter{chapter}{0}
                \stepcounter{section}
                \setcounter{section}{0}
                \setcounter{equation}{0}
                \setcounter{figure}{0}
                \setcounter{table}{0}
                \setcounter{footnote}{0}
  \def\@chapapp{\appendixname}%
  \renewcommand\thechapter{\@Alph\c@chapter}}

%  definition of sections
%  \hyphenpenalty and \raggedright added, so that there is no
%  hyphenation and the text is set ragged-right in sectioning

\def\runinsep{}
\def\aftertext{\unskip\runinsep}
%
\def\thesection{\thechapter.\arabic{section}}
\def\thesubsection{\thesection.\arabic{subsection}}
\def\thesubsubsection{\thesubsection.\arabic{subsubsection}}
\def\theparagraph{\thesubsubsection.\arabic{paragraph}}
\def\thesubparagraph{\theparagraph.\arabic{subparagraph}}
\def\chaptermark#1{}
%
\def\@ssect#1#2#3#4#5{%
  \@tempskipa #3\relax
  \ifdim \@tempskipa>\z@
    \begingroup
      #4{%
        \@hangfrom{\hskip #1}%
          \raggedright
          \hyphenpenalty \@M
          \interlinepenalty \@M #5\@@par}%
    \endgroup
  \else
    \def\@svsechd{#4{\hskip #1\relax #5}}%
  \fi
  \@xsect{#3}}
%
\def\@sect#1#2#3#4#5#6[#7]#8{%
   \ifnum #2>\c@secnumdepth
      \let\@svsec\@empty
   \else
      \refstepcounter{#1}%
      \protected@edef\@svsec{\@seccntformat{#1}\relax}%
   \fi
   \@tempskipa #5\relax
   \ifdim \@tempskipa>\z@
      \begingroup #6\relax
         \sec@hangfrom{\hskip #3\relax\@svsec}%
         {\raggedright
          \hyphenpenalty \@M
          \interlinepenalty \@M #8\@@par}%
      \endgroup
      \csname #1mark\endcsname{#7\seccounterend}%
      \addcontentsline{toc}{#1}{\ifnum #2>\c@secnumdepth
                                \else
                                   \protect\numberline{\csname the#1\endcsname\seccounterend}%
                                \fi
                                #7}%
   \else
      \def\@svsechd{%
         #6\hskip #3\relax
         \@svsec #8\aftertext\ignorespaces
         \csname #1mark\endcsname{#7}%
         \addcontentsline{toc}{#1}{%
            \ifnum #2>\c@secnumdepth \else
                \protect\numberline{\csname the#1\endcsname\seccounterend}%
            \fi
            #7}}%
   \fi
   \@xsect{#5}}

% figures and tables are processed in small print
\def \@floatboxreset {%
        \reset@font
        \small
        \@setnobreak
        \@setminipage
}
\def\fps@figure{htbp}
\def\fps@table{htbp}

% Frame for paste-in figures or tables
\def\mpicplace#1#2{%  #1 =width   #2 =height
\vbox{\hbox to #1{\vrule\@width \fboxrule \@height #2\hfill}}}

% labels of enumerate
\renewcommand\labelenumii{\theenumii)}
\renewcommand\theenumii{\@alph\c@enumii}

% labels of itemize
\renewcommand\labelitemi{\textbullet}
\renewcommand\labelitemii{\textendash}
\let\labelitemiii=\labelitemiv

% labels of description
\renewcommand*\descriptionlabel[1]{\hspace\labelsep #1\hfil}

% fixed indentation for standard itemize-environment
\newdimen\svitemindent \setlength{\svitemindent}{\parindent}


% make indentations changeable

\def\setitemindent#1{\settowidth{\labelwidth}{#1}%
        \let\setit@m=Y%
        \leftmargini\labelwidth
        \advance\leftmargini\labelsep
   \def\@listi{\leftmargin\leftmargini
        \labelwidth\leftmargini\advance\labelwidth by -\labelsep
        \parsep=\parskip
        \topsep=\medskipamount
        \itemsep=\parskip \advance\itemsep by -\parsep}}
\def\setitemitemindent#1{\settowidth{\labelwidth}{#1}%
        \let\setit@m=Y%
        \leftmarginii\labelwidth
        \advance\leftmarginii\labelsep
\def\@listii{\leftmargin\leftmarginii
        \labelwidth\leftmarginii\advance\labelwidth by -\labelsep
        \parsep=\parskip
        \topsep=\z@
        \itemsep=\parskip \advance\itemsep by -\parsep}}
%
% adjusted environment "description"
% if an optional parameter (at the first two levels of lists)
% is present, its width is considered to be the widest mark
% throughout the current list.
\def\description{\@ifnextchar[{\@describe}{\list{}{\labelwidth\z@
          \itemindent-\leftmargin \let\makelabel\descriptionlabel}}}
%
\def\describelabel#1{#1\hfil}
\def\@describe[#1]{\relax\ifnum\@listdepth=0
\setitemindent{#1}\else\ifnum\@listdepth=1
\setitemitemindent{#1}\fi\fi
\list{--}{\let\makelabel\describelabel}}
%
\def\itemize{%
  \ifnum \@itemdepth >\thr@@\@toodeep\else
    \advance\@itemdepth\@ne
    \ifx\setit@m\undefined
       \ifnum \@itemdepth=1 \leftmargini=\svitemindent
          \labelwidth\leftmargini\advance\labelwidth-\labelsep
          \leftmarginii=\leftmargini \leftmarginiii=\leftmargini
       \fi
    \fi
    \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
    \expandafter\list
      \csname\@itemitem\endcsname
      {\def\makelabel##1{\rlap{##1}\hss}}%
  \fi}
%
\newdimen\verbatimindent \verbatimindent\parindent
\def\verbatim{\advance\@totalleftmargin by\verbatimindent
\@verbatim \frenchspacing\@vobeyspaces \@xverbatim}

%
%  special signs and characters
\newcommand{\D}{\mathrm{d}}
\newcommand{\E}{\mathrm{e}}
\let\eul=\E
\newcommand{\I}{{\rm i}}
\let\imag=\I
%
% the definition of uppercase Greek characters
% Springer likes them as italics to depict variables
\DeclareMathSymbol{\Gamma}{\mathalpha}{letters}{"00}
\DeclareMathSymbol{\Delta}{\mathalpha}{letters}{"01}
\DeclareMathSymbol{\Theta}{\mathalpha}{letters}{"02}
\DeclareMathSymbol{\Lambda}{\mathalpha}{letters}{"03}
\DeclareMathSymbol{\Xi}{\mathalpha}{letters}{"04}
\DeclareMathSymbol{\Pi}{\mathalpha}{letters}{"05}
\DeclareMathSymbol{\Sigma}{\mathalpha}{letters}{"06}
\DeclareMathSymbol{\Upsilon}{\mathalpha}{letters}{"07}
\DeclareMathSymbol{\Phi}{\mathalpha}{letters}{"08}
\DeclareMathSymbol{\Psi}{\mathalpha}{letters}{"09}
\DeclareMathSymbol{\Omega}{\mathalpha}{letters}{"0A}
% the upright forms are defined here as \var<Character>
\DeclareMathSymbol{\varGamma}{\mathalpha}{operators}{"00}
\DeclareMathSymbol{\varDelta}{\mathalpha}{operators}{"01}
\DeclareMathSymbol{\varTheta}{\mathalpha}{operators}{"02}
\DeclareMathSymbol{\varLambda}{\mathalpha}{operators}{"03}
\DeclareMathSymbol{\varXi}{\mathalpha}{operators}{"04}
\DeclareMathSymbol{\varPi}{\mathalpha}{operators}{"05}
\DeclareMathSymbol{\varSigma}{\mathalpha}{operators}{"06}
\DeclareMathSymbol{\varUpsilon}{\mathalpha}{operators}{"07}
\DeclareMathSymbol{\varPhi}{\mathalpha}{operators}{"08}
\DeclareMathSymbol{\varPsi}{\mathalpha}{operators}{"09}
\DeclareMathSymbol{\varOmega}{\mathalpha}{operators}{"0A}
% Upright Lower Case Greek letters without using a new MathAlphabet
\newcommand{\greeksym}[1]{\usefont{U}{psy}{m}{n}#1}
\newcommand{\greeksymbold}[1]{{\usefont{U}{psy}{b}{n}#1}}
\newcommand{\allmodesymb}[2]{\relax\ifmmode{\mathchoice
{\mbox{\fontsize{\tf@size}{\tf@size}#1{#2}}}
{\mbox{\fontsize{\tf@size}{\tf@size}#1{#2}}}
{\mbox{\fontsize{\sf@size}{\sf@size}#1{#2}}}
{\mbox{\fontsize{\ssf@size}{\ssf@size}#1{#2}}}}
\else
\mbox{#1{#2}}\fi}
% Definition of lower case Greek letters
\newcommand{\ualpha}{\allmodesymb{\greeksym}{a}}
\newcommand{\ubeta}{\allmodesymb{\greeksym}{b}}
\newcommand{\uchi}{\allmodesymb{\greeksym}{c}}
\newcommand{\udelta}{\allmodesymb{\greeksym}{d}}
\newcommand{\ugamma}{\allmodesymb{\greeksym}{g}}
\newcommand{\umu}{\allmodesymb{\greeksym}{m}}
\newcommand{\unu}{\allmodesymb{\greeksym}{n}}
\newcommand{\upi}{\allmodesymb{\greeksym}{p}}
\newcommand{\utau}{\allmodesymb{\greeksym}{t}}
% redefines the \vec accent to a bold character - if desired
\def\fig@type{arrow}% temporarily abused
\ifx\vec@style\fig@type\else
\@ifundefined{vec@style}{%
 \def\vec#1{\ensuremath{\mathchoice
                     {\mbox{\boldmath$\displaystyle\mathbf{#1}$}}
                     {\mbox{\boldmath$\textstyle\mathbf{#1}$}}
                     {\mbox{\boldmath$\scriptstyle\mathbf{#1}$}}
                     {\mbox{\boldmath$\scriptscriptstyle\mathbf{#1}$}}}}%
}
{\def\vec#1{\ensuremath{\mathchoice
                     {\mbox{\boldmath$\displaystyle#1$}}
                     {\mbox{\boldmath$\textstyle#1$}}
                     {\mbox{\boldmath$\scriptstyle#1$}}
                     {\mbox{\boldmath$\scriptscriptstyle#1$}}}}%
}
\fi
% tensor
\def\tens#1{\relax\ifmmode\mathsf{#1}\else\textsf{#1}\fi}

% end of proof symbol
\newcommand\qedsymbol{\hbox{\rlap{$\sqcap$}$\sqcup$}}
\newcommand\qed{\relax\ifmmode\else\unskip\quad\fi\qedsymbol}
\newcommand\smartqed{\renewcommand\qed{\relax\ifmmode\qedsymbol\else
  {\unskip\nobreak\hfil\penalty50\hskip1em\null\nobreak\hfil\qedsymbol
  \parfillskip=\z@\finalhyphendemerits=0\endgraf}\fi}}
%
\def\num@book{%
\renewcommand\thesection{\thechapter.\@arabic\c@section}%
\renewcommand\thesubsection{\thesection.\@arabic\c@subsection}%
\renewcommand\theequation{\thechapter.\@arabic\c@equation}%
\renewcommand\thefigure{\thechapter.\@arabic\c@figure}%
\renewcommand\thetable{\thechapter.\@arabic\c@table}%
\@addtoreset{section}{chapter}%
\@addtoreset{figure}{chapter}%
\@addtoreset{table}{chapter}%
\@addtoreset{equation}{chapter}}
%
% Ragged bottom for the actual page
\def\thisbottomragged{\def\@textbottom{\vskip\z@ \@plus.0001fil
\global\let\@textbottom\relax}}

% This is texte.tex
% it defines various texts and their translations
% called up with documentstyle options
\def\switcht@vietnamese{%
\def\abstractname{Tóm tắt.}%
\def\ackname{Acknowledgement.}%
\def\andname{and}%
\def\bibname{Tài liệu tham khảo}%
\def\lastandname{, and}%
\def\appendixname{Phụ lục}%
\def\chaptername{Chương}%
\def\claimname{Claim}%
\def\conjecturename{Conjecture}%
\def\contentsname{Mục lục}%
\def\corollaryname{Corollary}%
\def\definitionname{Định nghĩa}%
\def\examplename{Ví dụ}%
\def\exercisename{Bài tập}%
\def\figurename{Hình}%
\def\keywordname{{\bf Key words:}}%
\def\indexname{Index}%
\def\lemmaname{Bổ đề}%
\def\contriblistname{List of Contributors}%
\def\listfigurename{List of Figures}%
\def\listtablename{List of Tables}%
\def\mailname{{\it Correspondence to\/}:}%
\def\noteaddname{Note added in proof}%
\def\notename{Note}%
\def\partname{Phần}%
\def\prefacename{Preface}%
\def\problemname{Problem}%
\def\proofname{Chứng minh}%
\def\propertyname{Property}%
\def\propositionname{Proposition}%
\def\questionname{Question}%
\def\refname{Tài liệu tham khảo}%
\def\remarkname{Remark}%
\def\seename{see}%
\def\solutionname{Solution}%
\def\subclassname{{\it Subject Classifications\/}:}%
\def\tablename{Table}%
\def\theoremname{Theorem}}
\switcht@vietnamese
% Names of theorem like environments are already defined
% but must be translated if another language is chosen
%
% French section
\def\switcht@francais{\svlanginfo
 \def\abstractname{R\'esum\'e.}%
 \def\ackname{Remerciements.}%
 \def\andname{et}%
 \def\lastandname{ et}%
 \def\appendixname{Appendice}%
 \def\bibname{Bibliographie}%
 \def\chaptername{Chapitre}%
 \def\claimname{Pr\'etention}%
 \def\conjecturename{Hypoth\`ese}%
 \def\contentsname{Table des mati\`eres}%
 \def\corollaryname{Corollaire}%
 \def\definitionname{D\'efinition}%
 \def\examplename{Exemple}%
 \def\exercisename{Exercice}%
 \def\figurename{Fig.}%
 \def\keywordname{{\bf Mots-cl\'e:}}%
 \def\indexname{Index}%
 \def\lemmaname{Lemme}%
 \def\contriblistname{Liste des contributeurs}%
 \def\listfigurename{Liste des figures}%
 \def\listtablename{Liste des tables}%
 \def\mailname{{\it Correspondence to\/}:}%
 \def\noteaddname{Note ajout\'ee \`a l'\'epreuve}%
 \def\notename{Remarque}%
 \def\partname{Partie}%
 \def\prefacename{Avant-propos}%  ou Pr\'eface
 \def\problemname{Probl\`eme}%
 \def\proofname{Preuve}%
 \def\propertyname{Caract\'eristique}%
%\def\propositionname{Proposition}%
 \def\questionname{Question}%
 \def\refname{Litt\'erature}%
 \def\remarkname{Remarque}%
 \def\seename{voir}%
 \def\solutionname{Solution}%
 \def\subclassname{{\it Subject Classifications\/}:}%
 \def\tablename{Tableau}%
 \def\theoremname{Th\'eor\`eme}%
}
%
% German section
\def\switcht@deutsch{\svlanginfo
 \def\abstractname{Zusammenfassung.}%
 \def\ackname{Danksagung.}%
 \def\andname{und}%
 \def\lastandname{ und}%
 \def\appendixname{Anhang}%
 \def\bibname{Literaturverzeichnis}%
 \def\chaptername{Kapitel}%
 \def\claimname{Behauptung}%
 \def\conjecturename{Hypothese}%
 \def\contentsname{Inhaltsverzeichnis}%
 \def\corollaryname{Korollar}%
%\def\definitionname{Definition}%
 \def\examplename{Beispiel}%
 \def\exercisename{\"Ubung}%
 \def\figurename{Abb.}%
 \def\keywordname{{\bf Schl\"usselw\"orter:}}%
 \def\indexname{Sachverzeichnis}%
%\def\lemmaname{Lemma}%
 \def\contriblistname{Mitarbeiter}%
 \def\listfigurename{Abbildungsverzeichnis}%
 \def\listtablename{Tabellenverzeichnis}%
 \def\mailname{{\it Correspondence to\/}:}%
 \def\noteaddname{Nachtrag}%
 \def\notename{Anmerkung}%
 \def\partname{Teil}%
 \def\prefacename{Vorwort}%
%\def\problemname{Problem}%
 \def\proofname{Beweis}%
 \def\propertyname{Eigenschaft}%
%\def\propositionname{Proposition}%
 \def\questionname{Frage}%
 \def\refname{Literaturverzeichnis}%
 \def\remarkname{Anmerkung}%
 \def\seename{siehe}%
 \def\solutionname{L\"osung}%
 \def\subclassname{{\it Subject Classifications\/}:}%
 \def\tablename{Tabelle}%
%\def\theoremname{Theorem}%
}
%
% Vietnamese section
\def\switcht@vietnamese{\svlanginfo
 \def\abstractname{Zusammenfassung.}%
 \def\ackname{Danksagung.}%
 \def\andname{und}%
 \def\lastandname{ und}%
 \def\appendixname{Anhang}%
 \def\bibname{Literaturverzeichnis}%
 \def\chaptername{Kapitel}%
 \def\claimname{Behauptung}%
 \def\conjecturename{Hypothese}%
 \def\contentsname{Inhaltsverzeichnis}%
 \def\corollaryname{Korollar}%
%\def\definitionname{Definition}%
 \def\examplename{Beispiel}%
 \def\exercisename{\"Ubung}%
 \def\figurename{Abb.}%
 \def\keywordname{{\bf Schl\"usselw\"orter:}}%
 \def\indexname{Sachverzeichnis}%
%\def\lemmaname{Lemma}%
 \def\contriblistname{Mitarbeiter}%
 \def\listfigurename{Abbildungsverzeichnis}%
 \def\listtablename{Tabellenverzeichnis}%
 \def\mailname{{\it Correspondence to\/}:}%
 \def\noteaddname{Nachtrag}%
 \def\notename{Anmerkung}%
 \def\partname{Teil}%
 \def\prefacename{Vorwort}%
%\def\problemname{Problem}%
 \def\proofname{Beweis}%
 \def\propertyname{Eigenschaft}%
%\def\propositionname{Proposition}%
 \def\questionname{Frage}%
 \def\refname{Literaturverzeichnis}%
 \def\remarkname{Anmerkung}%
 \def\seename{siehe}%
 \def\solutionname{L\"osung}%
 \def\subclassname{{\it Subject Classifications\/}:}%
 \def\tablename{Tabelle}%
%\def\theoremname{Theorem}%
}
\def\getsto{\mathrel{\mathchoice {\vcenter{\offinterlineskip
\halign{\hfil
$\displaystyle##$\hfil\cr\gets\cr\to\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\textstyle##$\hfil\cr\gets
\cr\to\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptstyle##$\hfil\cr\gets
\cr\to\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptscriptstyle##$\hfil\cr
\gets\cr\to\cr}}}}}
\def\lid{\mathrel{\mathchoice {\vcenter{\offinterlineskip\halign{\hfil
$\displaystyle##$\hfil\cr<\cr\noalign{\vskip1.2\p@}=\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\textstyle##$\hfil\cr<\cr
\noalign{\vskip1.2\p@}=\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptstyle##$\hfil\cr<\cr
\noalign{\vskip\p@}=\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptscriptstyle##$\hfil\cr
<\cr
\noalign{\vskip0.9\p@}=\cr}}}}}
\def\gid{\mathrel{\mathchoice {\vcenter{\offinterlineskip\halign{\hfil
$\displaystyle##$\hfil\cr>\cr\noalign{\vskip1.2\p@}=\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\textstyle##$\hfil\cr>\cr
\noalign{\vskip1.2\p@}=\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptstyle##$\hfil\cr>\cr
\noalign{\vskip\p@}=\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptscriptstyle##$\hfil\cr
>\cr
\noalign{\vskip0.9\p@}=\cr}}}}}
\def\grole{\mathrel{\mathchoice {\vcenter{\offinterlineskip
\halign{\hfil
$\displaystyle##$\hfil\cr>\cr\noalign{\vskip-\p@}<\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\textstyle##$\hfil\cr
>\cr\noalign{\vskip-\p@}<\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptstyle##$\hfil\cr
>\cr\noalign{\vskip-0.8\p@}<\cr}}}
{\vcenter{\offinterlineskip\halign{\hfil$\scriptscriptstyle##$\hfil\cr
>\cr\noalign{\vskip-0.3\p@}<\cr}}}}}
\def\bbbr{{\rm I\!R}} %reelle Zahlen
\def\bbbm{{\rm I\!M}}
\def\bbbn{{\rm I\!N}} %natuerliche Zahlen
\def\bbbf{{\rm I\!F}}
\def\bbbh{{\rm I\!H}}
\def\bbbk{{\rm I\!K}}
\def\bbbp{{\rm I\!P}}
\def\bbbone{{\mathchoice {\rm 1\mskip-4mu l} {\rm 1\mskip-4mu l}
{\rm 1\mskip-4.5mu l} {\rm 1\mskip-5mu l}}}
\def\bbbc{{\mathchoice {\setbox0=\hbox{$\displaystyle\rm C$}\hbox{\hbox
to\z@{\kern0.4\wd0\vrule\@height0.9\ht0\hss}\box0}}
{\setbox0=\hbox{$\textstyle\rm C$}\hbox{\hbox
to\z@{\kern0.4\wd0\vrule\@height0.9\ht0\hss}\box0}}
{\setbox0=\hbox{$\scriptstyle\rm C$}\hbox{\hbox
to\z@{\kern0.4\wd0\vrule\@height0.9\ht0\hss}\box0}}
{\setbox0=\hbox{$\scriptscriptstyle\rm C$}\hbox{\hbox
to\z@{\kern0.4\wd0\vrule\@height0.9\ht0\hss}\box0}}}}
\def\bbbq{{\mathchoice {\setbox0=\hbox{$\displaystyle\rm
Q$}\hbox{\raise
0.15\ht0\hbox to\z@{\kern0.4\wd0\vrule\@height0.8\ht0\hss}\box0}}
{\setbox0=\hbox{$\textstyle\rm Q$}\hbox{\raise
0.15\ht0\hbox to\z@{\kern0.4\wd0\vrule\@height0.8\ht0\hss}\box0}}
{\setbox0=\hbox{$\scriptstyle\rm Q$}\hbox{\raise
0.15\ht0\hbox to\z@{\kern0.4\wd0\vrule\@height0.7\ht0\hss}\box0}}
{\setbox0=\hbox{$\scriptscriptstyle\rm Q$}\hbox{\raise
0.15\ht0\hbox to\z@{\kern0.4\wd0\vrule\@height0.7\ht0\hss}\box0}}}}
\def\bbbt{{\mathchoice {\setbox0=\hbox{$\displaystyle\rm
T$}\hbox{\hbox to\z@{\kern0.3\wd0\vrule\@height0.9\ht0\hss}\box0}}
{\setbox0=\hbox{$\textstyle\rm T$}\hbox{\hbox
to\z@{\kern0.3\wd0\vrule\@height0.9\ht0\hss}\box0}}
{\setbox0=\hbox{$\scriptstyle\rm T$}\hbox{\hbox
to\z@{\kern0.3\wd0\vrule\@height0.9\ht0\hss}\box0}}
{\setbox0=\hbox{$\scriptscriptstyle\rm T$}\hbox{\hbox
to\z@{\kern0.3\wd0\vrule\@height0.9\ht0\hss}\box0}}}}
\def\bbbs{{\mathchoice
{\setbox0=\hbox{$\displaystyle     \rm S$}\hbox{\raise0.5\ht0\hbox
to\z@{\kern0.35\wd0\vrule\@height0.45\ht0\hss}\hbox
to\z@{\kern0.55\wd0\vrule\@height0.5\ht0\hss}\box0}}
{\setbox0=\hbox{$\textstyle        \rm S$}\hbox{\raise0.5\ht0\hbox
to\z@{\kern0.35\wd0\vrule\@height0.45\ht0\hss}\hbox
to\z@{\kern0.55\wd0\vrule\@height0.5\ht0\hss}\box0}}
{\setbox0=\hbox{$\scriptstyle      \rm S$}\hbox{\raise0.5\ht0\hbox
to\z@{\kern0.35\wd0\vrule\@height0.45\ht0\hss}\raise0.05\ht0\hbox
to\z@{\kern0.5\wd0\vrule\@height0.45\ht0\hss}\box0}}
{\setbox0=\hbox{$\scriptscriptstyle\rm S$}\hbox{\raise0.5\ht0\hbox
to\z@{\kern0.4\wd0\vrule\@height0.45\ht0\hss}\raise0.05\ht0\hbox
to\z@{\kern0.55\wd0\vrule\@height0.45\ht0\hss}\box0}}}}
\def\bbbz{{\mathchoice {\hbox{$\textstyle\sf Z\kern-0.4em Z$}}
{\hbox{$\textstyle\sf Z\kern-0.4em Z$}}
{\hbox{$\scriptstyle\sf Z\kern-0.3em Z$}}
{\hbox{$\scriptscriptstyle\sf Z\kern-0.2em Z$}}}}

\let\ts\,

\setlength \labelsep     {5\p@}
\setlength\leftmargini   {17\p@}
\setlength\leftmargin    {\leftmargini}
\setlength\leftmarginii  {\leftmargini}
\setlength\leftmarginiii {\leftmargini}
\setlength\leftmarginiv  {\leftmargini}
\setlength\labelwidth    {\leftmargini}
\addtolength\labelwidth{-\labelsep}

\def\@listI{\leftmargin\leftmargini
        \parsep=\parskip
        \topsep=\medskipamount
        \itemsep=\parskip \advance\itemsep by -\parsep}
\let\@listi\@listI
\@listi

\def\@listii{\leftmargin\leftmarginii
        \labelwidth\leftmarginii
        \advance\labelwidth by -\labelsep
        \parsep=\parskip
        \topsep=\z@
        \itemsep=\parskip
        \advance\itemsep by -\parsep}

\def\@listiii{\leftmargin\leftmarginiii
        \labelwidth\leftmarginiii\advance\labelwidth by -\labelsep
        \parsep=\parskip
        \topsep=\z@
        \itemsep=\parskip
        \advance\itemsep by -\parsep
        \partopsep=\topsep}

\setlength\arraycolsep{1.5\p@}
\setlength\tabcolsep{1.5\p@}

\def\tableofcontents{\@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
 \fi\chapter*{\contentsname \@mkboth{{\contentsname}}{{\contentsname}}}
 \@starttoc{toc}\if@restonecol\twocolumn\fi}

\setcounter{tocdepth}{0}

\def\l@part#1#2{\addpenalty{\@secpenalty}%
   \addvspace{2em \@plus\p@}%
   \begingroup
     \parindent \z@
     \rightskip \z@ \@plus 5em
     \hrule\vskip5\p@
     \bfseries\boldmath
     \leavevmode
     #1\par
     \vskip5\p@
     \hrule
     \vskip\p@
     \nobreak
   \endgroup}

\def\@dotsep{2}

\def\addnumcontentsmark#1#2#3{%
\addtocontents{#1}{\protect\contentsline{#2}{\protect\numberline
                                    {\thechapter}#3}{\thepage}}}
\def\addcontentsmark#1#2#3{%
\addtocontents{#1}{\protect\contentsline{#2}{#3}{\thepage}}}
\def\addcontentsmarkwop#1#2#3{%
\addtocontents{#1}{\protect\contentsline{#2}{#3}{0}}}

\def\@adcmk[#1]{\ifcase #1 \or
\def\@gtempa{\addnumcontentsmark}%
  \or    \def\@gtempa{\addcontentsmark}%
  \or    \def\@gtempa{\addcontentsmarkwop}%
  \fi\@gtempa{toc}{chapter}}
\def\addtocmark{\@ifnextchar[{\@adcmk}{\@adcmk[3]}}

\def\l@chapter#1#2{\par\addpenalty{-\@highpenalty}
 \addvspace{1.0em \@plus \p@}
 \@tempdima \tocchpnum \begingroup
 \parindent \z@ \rightskip \@tocrmarg
 \advance\rightskip by \z@ \@plus 2cm
 \parfillskip -\rightskip \pretolerance=10000
 \leavevmode \advance\leftskip\@tempdima \hskip -\leftskip
 {\bfseries\boldmath#1}\ifx0#2\hfil\null
 \else
    \nobreak
    \leaders\hbox{$\m@th \mkern \@dotsep mu\hbox{.}\mkern
    \@dotsep mu$}\hfill
    \nobreak\hbox to\@pnumwidth{\hfil #2}%
 \fi\par
 \penalty\@highpenalty \endgroup}

\newdimen\tocchpnum
\newdimen\tocsecnum
\newdimen\tocsectotal
\newdimen\tocsubsecnum
\newdimen\tocsubsectotal
\newdimen\tocsubsubsecnum
\newdimen\tocsubsubsectotal
\newdimen\tocparanum
\newdimen\tocparatotal
\newdimen\tocsubparanum
\tocchpnum=20\p@            % chapter {\bf 88.} \@plus 5.3\p@
\tocsecnum=22.5\p@          % section 88.8. plus 4.722\p@
\tocsubsecnum=30.5\p@       % subsection 88.8.8 plus 4.944\p@
\tocsubsubsecnum=38\p@      % subsubsection 88.8.8.8 plus 4.666\p@
\tocparanum=45\p@           % paragraph 88.8.8.8.8 plus 3.888\p@
\tocsubparanum=53\p@        % subparagraph 88.8.8.8.8.8 plus 4.11\p@
\def\calctocindent{%
\tocsectotal=\tocchpnum
\advance\tocsectotal by\tocsecnum
\tocsubsectotal=\tocsectotal
\advance\tocsubsectotal by\tocsubsecnum
\tocsubsubsectotal=\tocsubsectotal
\advance\tocsubsubsectotal by\tocsubsubsecnum
\tocparatotal=\tocsubsubsectotal
\advance\tocparatotal by\tocparanum}
\calctocindent

\def\@dottedtocline#1#2#3#4#5{%
  \ifnum #1>\c@tocdepth \else
    \vskip \z@ \@plus.2\p@
    {\leftskip #2\relax \rightskip \@tocrmarg \advance\rightskip by \z@ \@plus 2cm
               \parfillskip -\rightskip \pretolerance=10000
     \parindent #2\relax\@afterindenttrue
     \interlinepenalty\@M
     \leavevmode
     \@tempdima #3\relax
     \advance\leftskip \@tempdima \null\nobreak\hskip -\leftskip
     {#4}\nobreak
     \leaders\hbox{$\m@th
        \mkern \@dotsep mu\hbox{.}\mkern \@dotsep
        mu$}\hfill
     \nobreak
     \hb@xt@\@pnumwidth{\hfil\normalfont \normalcolor #5}%
     \par}%
  \fi}
%
% \def\l@section{\@dottedtocline{1}{\tocchpnum}{\tocsecnum}}
% \def\l@subsection{\@dottedtocline{2}{\tocsectotal}{\tocsubsecnum}}
% \def\l@subsubsection{\@dottedtocline{3}{\tocsubsectotal}{\tocsubsubsecnum}}
% \def\l@paragraph{\@dottedtocline{4}{\tocsubsubsectotal}{\tocparanum}}
% \def\l@subparagraph{\@dottedtocline{5}{\tocparatotal}{\tocsubparanum}}

\def\l@section{\@dottedtocline{1}{1.5em}{2.3em}}
\def\l@subsection{\@dottedtocline{2}{3.8em}{3.2em}}
\def\l@subsubsection{\@dottedtocline{3}{7.0em}{4.1em}}
\def\l@paragraph{\@dottedtocline{4}{10em}{5em}}
\def\l@subparagraph{\@dottedtocline{5}{12em}{6em}}


\renewcommand\listoffigures{%
    \chapter*{\listfigurename
      \@mkboth{\listfigurename}{\listfigurename}}%
    \@starttoc{lof}%
    }

\renewcommand\listoftables{%
    \chapter*{\listtablename
      \@mkboth{\listtablename}{\listtablename}}%
    \@starttoc{lot}%
    }

\renewcommand\footnoterule{%
  \kern-3\p@
  \hrule\@width 50\p@
  \kern2.6\p@}

\newdimen\foot@parindent
\foot@parindent 10.83\p@

\AtBeginDocument{%
\long\def\@makefntext#1{\@setpar{\@@par\@tempdima \hsize
         \advance\@tempdima-\foot@parindent\parshape\@ne\foot@parindent
         \@tempdima}\par
         \parindent \foot@parindent\noindent \hbox to \z@{%
         \hss\hss$^{\@thefnmark}$ }#1}}

\if@spthms
% Definition of the "\spnewtheorem" command.
%
% Usage:
%
%     \spnewtheorem{env_nam}{caption}[within]{cap_font}{body_font}
% or  \spnewtheorem{env_nam}[numbered_like]{caption}{cap_font}{body_font}
% or  \spnewtheorem*{env_nam}{caption}{cap_font}{body_font}
%
% New is "cap_font" and "body_font". It stands for
% fontdefinition of the caption and the text itself.
%
% "\spnewtheorem*" gives a theorem without number.
%
% A defined spnewthoerem environment is used as described
% by Lamport.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\def\@thmcountersep{.}
\def\@thmcounterend{.}
\newcommand\nocaption{\noexpand\@gobble}
\newdimen\spthmsep \spthmsep=3pt

\def\spnewtheorem{\@ifstar{\@sthm}{\@Sthm}}

% definition of \spnewtheorem with number

\def\@spnthm#1#2{%
  \@ifnextchar[{\@spxnthm{#1}{#2}}{\@spynthm{#1}{#2}}}
\def\@Sthm#1{\@ifnextchar[{\@spothm{#1}}{\@spnthm{#1}}}

\def\@spxnthm#1#2[#3]#4#5{\expandafter\@ifdefinable\csname #1\endcsname
   {\@definecounter{#1}\@addtoreset{#1}{#3}%
   \expandafter\xdef\csname the#1\endcsname{\expandafter\noexpand
     \csname the#3\endcsname \noexpand\@thmcountersep \@thmcounter{#1}}%
   \expandafter\xdef\csname #1name\endcsname{#2}%
   \global\@namedef{#1}{\@spthm{#1}{\csname #1name\endcsname}{#4}{#5}}%
                              \global\@namedef{end#1}{\@endtheorem}}}

\def\@spynthm#1#2#3#4{\expandafter\@ifdefinable\csname #1\endcsname
   {\@definecounter{#1}%
   \expandafter\xdef\csname the#1\endcsname{\@thmcounter{#1}}%
   \expandafter\xdef\csname #1name\endcsname{#2}%
   \global\@namedef{#1}{\@spthm{#1}{\csname #1name\endcsname}{#3}{#4}}%
                               \global\@namedef{end#1}{\@endtheorem}}}

\def\@spothm#1[#2]#3#4#5{%
  \@ifundefined{c@#2}{\@latexerr{No theorem environment `#2' defined}\@eha}%
  {\expandafter\@ifdefinable\csname #1\endcsname
  {\global\@namedef{the#1}{\@nameuse{the#2}}%
  \expandafter\xdef\csname #1name\endcsname{#3}%
  \global\@namedef{#1}{\@spthm{#2}{\csname #1name\endcsname}{#4}{#5}}%
  \global\@namedef{end#1}{\@endtheorem}}}}

\def\@spthm#1#2#3#4{\topsep 7\p@ \@plus2\p@ \@minus4\p@
\labelsep=\spthmsep\refstepcounter{#1}%
\@ifnextchar[{\@spythm{#1}{#2}{#3}{#4}}{\@spxthm{#1}{#2}{#3}{#4}}}

\def\@spxthm#1#2#3#4{\@spbegintheorem{#2}{\csname the#1\endcsname}{#3}{#4}%
                    \ignorespaces}

\def\@spythm#1#2#3#4[#5]{\@spopargbegintheorem{#2}{\csname
       the#1\endcsname}{#5}{#3}{#4}\ignorespaces}

\def\normalthmheadings{\def\@spbegintheorem##1##2##3##4{\trivlist
                 \item[\hskip\labelsep{##3##1\ ##2\@thmcounterend}]##4}
\def\@spopargbegintheorem##1##2##3##4##5{\trivlist
      \item[\hskip\labelsep{##4##1\ ##2}]{##4(##3)\@thmcounterend\ }##5}}
\normalthmheadings

\def\reversethmheadings{\def\@spbegintheorem##1##2##3##4{\trivlist
                 \item[\hskip\labelsep{##3##2\ ##1\@thmcounterend}]##4}
\def\@spopargbegintheorem##1##2##3##4##5{\trivlist
      \item[\hskip\labelsep{##4##2\ ##1}]{##4(##3)\@thmcounterend\ }##5}}

% definition of \spnewtheorem* without number

\def\@sthm#1#2{\@Ynthm{#1}{#2}}

\def\@Ynthm#1#2#3#4{\expandafter\@ifdefinable\csname #1\endcsname
   {\global\@namedef{#1}{\@Thm{\csname #1name\endcsname}{#3}{#4}}%
    \expandafter\xdef\csname #1name\endcsname{#2}%
    \global\@namedef{end#1}{\@endtheorem}}}

\def\@Thm#1#2#3{\topsep 7\p@ \@plus2\p@ \@minus4\p@
\@ifnextchar[{\@Ythm{#1}{#2}{#3}}{\@Xthm{#1}{#2}{#3}}}

\def\@Xthm#1#2#3{\@Begintheorem{#1}{#2}{#3}\ignorespaces}

\def\@Ythm#1#2#3[#4]{\@Opargbegintheorem{#1}
       {#4}{#2}{#3}\ignorespaces}

\def\@Begintheorem#1#2#3{#3\trivlist
                           \item[\hskip\labelsep{#2#1\@thmcounterend}]}

\def\@Opargbegintheorem#1#2#3#4{#4\trivlist
      \item[\hskip\labelsep{#3#1}]{#3(#2)\@thmcounterend\ }}

% initialize theorem environment

\if@envcntshowhiercnt % show hierarchy counter
   \def\@thmcountersep{.}
   \spnewtheorem{theorem}{Theorem}[\envankh]{\bfseries}{\itshape}
   \@addtoreset{theorem}{chapter}
\else          % theorem counter only
   \spnewtheorem{theorem}{Theorem}{\bfseries}{\itshape}
   \if@envcntreset
      \@addtoreset{theorem}{chapter}
      \if@envcntresetsect
         \@addtoreset{theorem}{section}
      \fi
   \fi
\fi

%definition of divers theorem environments
\spnewtheorem*{claim}{Claim}{\itshape}{\rmfamily}
\spnewtheorem*{proof}{Proof}{\itshape}{\rmfamily}
%
\if@envcntsame % all environments like "Theorem" - using its counter
   \def\spn@wtheorem#1#2#3#4{\@spothm{#1}[theorem]{#2}{#3}{#4}}
\else % all environments with their own counter
   \if@envcntshowhiercnt % show hierarchy counter
      \def\spn@wtheorem#1#2#3#4{\@spxnthm{#1}{#2}[\envankh]{#3}{#4}}
   \else          % environment counter only
      \if@envcntreset % environment counter is reset each section
         \if@envcntresetsect
            \def\spn@wtheorem#1#2#3#4{\@spynthm{#1}{#2}{#3}{#4}
             \@addtoreset{#1}{chapter}\@addtoreset{#1}{section}}
         \else
            \def\spn@wtheorem#1#2#3#4{\@spynthm{#1}{#2}{#3}{#4}
                                      \@addtoreset{#1}{chapter}}
         \fi
      \else
         \let\spn@wtheorem=\@spynthm
      \fi
   \fi
\fi
%
\let\spdefaulttheorem=\spn@wtheorem
%
\spn@wtheorem{case}{Case}{\itshape}{\rmfamily}
\spn@wtheorem{conjecture}{Conjecture}{\itshape}{\rmfamily}
\spn@wtheorem{corollary}{Corollary}{\bfseries}{\itshape}
\spn@wtheorem{definition}{Definition}{\bfseries}{\itshape}
\spn@wtheorem{example}{Ví}{\itshape}{\rmfamily}
\spn@wtheorem{exercise}{Exercise}{\bfseries}{\rmfamily}
\spn@wtheorem{lemma}{Lemma}{\bfseries}{\itshape}
\spn@wtheorem{note}{Note}{\itshape}{\rmfamily}
\spn@wtheorem{problem}{Problem}{\bfseries}{\rmfamily}
\spn@wtheorem{property}{Property}{\itshape}{\rmfamily}
\spn@wtheorem{proposition}{Proposition}{\bfseries}{\itshape}
\spn@wtheorem{question}{Question}{\itshape}{\rmfamily}
\spn@wtheorem{solution}{Solution}{\bfseries}{\rmfamily}
\spn@wtheorem{remark}{Remark}{\itshape}{\rmfamily}
%
\newenvironment{theopargself}
    {\def\@spopargbegintheorem##1##2##3##4##5{\trivlist
         \item[\hskip\labelsep{##4##1\ ##2}]{##4##3\@thmcounterend\ }##5}
     \def\@Opargbegintheorem##1##2##3##4{##4\trivlist
         \item[\hskip\labelsep{##3##1}]{##3##2\@thmcounterend\ }}}{}
\newenvironment{theopargself*}
    {\def\@spopargbegintheorem##1##2##3##4##5{\trivlist
         \item[\hskip\labelsep{##4##1\ ##2}]{\hspace*{-\labelsep}##4##3\@thmcounterend}##5}
     \def\@Opargbegintheorem##1##2##3##4{##4\trivlist
         \item[\hskip\labelsep{##3##1}]{\hspace*{-\labelsep}##3##2\@thmcounterend}}}{}
%
\spnewtheorem{prob}{\nocaption}[chapter]{\bfseries}{\rmfamily}
\newcommand{\probref}[1]{\textbf{\ref{#1}} }
\newenvironment{sol}{\par\addvspace{6pt}\noindent\probref}{\par\addvspace{6pt}}
%
\fi

\def\@takefromreset#1#2{%
    \def\@tempa{#1}%
    \let\@tempd\@elt
    \def\@elt##1{%
        \def\@tempb{##1}%
        \ifx\@tempa\@tempb\else
            \@addtoreset{##1}{#2}%
        \fi}%
    \expandafter\expandafter\let\expandafter\@tempc\csname cl@#2\endcsname
    \expandafter\def\csname cl@#2\endcsname{}%
    \@tempc
    \let\@elt\@tempd}

% redefininition of the captions for "figure" and "table" environments
%
\@ifundefined{floatlegendstyle}{\def\floatlegendstyle{\bfseries}}{}
\def\floatcounterend{.\ }
\def\capstrut{\vrule\@width\z@\@height\topskip}
\@ifundefined{captionstyle}{\def\captionstyle{\normalfont\small}}{}
\@ifundefined{instindent}{\newdimen\instindent}{}

\long\def\@caption#1[#2]#3{\par\addcontentsline{\csname
  ext@#1\endcsname}{#1}{\protect\numberline{\csname
  the#1\endcsname}{\ignorespaces #2}}\begingroup
    \@parboxrestore\if@minipage\@setminipage\fi
    \@makecaption{\csname fnum@#1\endcsname}{\ignorespaces #3}\par
  \endgroup}

\def\twocaptionwidth#1#2{\def\first@capwidth{#1}\def\second@capwidth{#2}}
% Default: .46\textwidth
\twocaptionwidth{.46\textwidth}{.46\textwidth}

\def\leftcaption{\refstepcounter\@captype\@dblarg%
            {\@leftcaption\@captype}}

\def\rightcaption{\refstepcounter\@captype\@dblarg%
            {\@rightcaption\@captype}}

\long\def\@leftcaption#1[#2]#3{\addcontentsline{\csname
  ext@#1\endcsname}{#1}{\protect\numberline{\csname
  the#1\endcsname}{\ignorespaces #2}}\begingroup
    \@parboxrestore
    \vskip\figcapgap
    \@maketwocaptions{\csname fnum@#1\endcsname}{\ignorespaces #3}%
    {\first@capwidth}\ignorespaces\hspace{.073\textwidth}\hfill%
  \endgroup}

\long\def\@rightcaption#1[#2]#3{\addcontentsline{\csname
  ext@#1\endcsname}{#1}{\protect\numberline{\csname
  the#1\endcsname}{\ignorespaces #2}}\begingroup
    \@parboxrestore
    \@maketwocaptions{\csname fnum@#1\endcsname}{\ignorespaces #3}%
    {\second@capwidth}\par
  \endgroup}

\long\def\@maketwocaptions#1#2#3{%
   \parbox[t]{#3}{{\floatlegendstyle #1\floatcounterend}#2}}

\def\fig@pos{l}
\newcommand{\leftfigure}[2][\fig@pos]{\makebox[.4635\textwidth][#1]{#2}}
\let\rightfigure\leftfigure

\newdimen\figgap\figgap=0.5cm  % hgap between figure and sidecaption
%
\long\def\@makesidecaption#1#2{%
   \setbox0=\vbox{\hsize=\@tempdimb
                  \captionstyle{\floatlegendstyle
                                         #1\floatcounterend}#2}%
   \ifdim\instindent<\z@
      \ifdim\ht0>-\instindent
         \advance\instindent by\ht0
         \typeout{^^JClass-Warning: Legend of \string\sidecaption\space for
                     \@captype\space\csname the\@captype\endcsname
                  ^^Jis \the\instindent\space taller than the corresponding float -
                  ^^Jyou'd better switch the environment. }%
         \instindent\z@
      \fi
   \else
      \ifdim\ht0<\instindent
         \advance\instindent by-\ht0
         \advance\instindent by-\dp0\relax
         \advance\instindent by\topskip
         \advance\instindent by-11\p@
      \else
         \advance\instindent by-\ht0
         \instindent=-\instindent
         \typeout{^^JClass-Warning: Legend of \string\sidecaption\space for
                     \@captype\space\csname the\@captype\endcsname
                  ^^Jis \the\instindent\space taller than the corresponding float -
                  ^^Jyou'd better switch the environment. }%
         \instindent\z@
      \fi
   \fi
   \parbox[b]{\@tempdimb}{\captionstyle{\floatlegendstyle
                                        #1\floatcounterend}#2%
                          \ifdim\instindent>\z@ \\
                               \vrule\@width\z@\@height\instindent
                                     \@depth\z@
                          \fi}}
\def\sidecaption{\@ifnextchar[\sidec@ption{\sidec@ption[b]}}
\def\sidec@ption[#1]#2\caption{%
\setbox\@tempboxa=\hbox{\ignorespaces#2\unskip}%
\if@twocolumn
 \ifdim\hsize<\textwidth\else
   \ifdim\wd\@tempboxa<\columnwidth
      \typeout{Double column float fits into single column -
            ^^Jyou'd better switch the environment. }%
   \fi
 \fi
\fi
  \instindent=\ht\@tempboxa
  \advance\instindent by\dp\@tempboxa
\if t#1
\else
  \instindent=-\instindent
\fi
\@tempdimb=\hsize
\advance\@tempdimb by-\figgap
\advance\@tempdimb by-\wd\@tempboxa
\ifdim\@tempdimb<3cm
   \ClassWarning{SVMono}{\string\sidecaption: No sufficient room for the legend;
             ^^Jusing normal \string\caption}%
   \unhbox\@tempboxa
   \let\@capcommand=\@caption
\else
   \ifdim\@tempdimb<4.5cm
      \ClassWarning{SVMono}{\string\sidecaption: Room for the legend very narrow;
               ^^Jusing \string\raggedright}%
      \toks@\expandafter{\captionstyle\sloppy
                         \rightskip=\z@\@plus6mm\relax}%
      \def\captionstyle{\the\toks@}%
   \fi
   \let\@capcommand=\@sidecaption
   \leavevmode
   \unhbox\@tempboxa
   \hfill
\fi
\refstepcounter\@captype
\@dblarg{\@capcommand\@captype}}
\long\def\@sidecaption#1[#2]#3{\addcontentsline{\csname
  ext@#1\endcsname}{#1}{\protect\numberline{\csname
  the#1\endcsname}{\ignorespaces #2}}\begingroup
    \@parboxrestore
    \@makesidecaption{\csname fnum@#1\endcsname}{\ignorespaces #3}\par
  \endgroup}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\def\fig@type{figure}

\def\leftlegendglue{\hfil}
\newdimen\figcapgap\figcapgap=5\p@   % vgap between figure and caption
\newdimen\tabcapgap\tabcapgap=5.5\p@ % vgap between caption and table

\long\def\@makecaption#1#2{%
 \captionstyle
 \ifx\@captype\fig@type
   \vskip\figcapgap
 \fi
 \setbox\@tempboxa\hbox{{\floatlegendstyle #1\floatcounterend}%
 \capstrut #2}%
 \ifdim \wd\@tempboxa >\hsize
   {\floatlegendstyle #1\floatcounterend}\capstrut #2\par
 \else
   \hbox to\hsize{\leftlegendglue\unhbox\@tempboxa\hfil}%
 \fi
 \ifx\@captype\fig@type\else
   \vskip\tabcapgap
 \fi}

\newcounter{merk}

\def\endfigure{\resetsubfig\end@float}

\@namedef{endfigure*}{\resetsubfig\end@dblfloat}

\def\resetsubfig{\global\let\last@subfig=\undefined}

\def\r@setsubfig{\xdef\last@subfig{\number\value{figure}}%
\setcounter{figure}{\value{merk}}%
\setcounter{merk}{0}}

\def\subfigures{\refstepcounter{figure}%
   \@tempcnta=\value{merk}%
   \setcounter{merk}{\value{figure}}%
   \setcounter{figure}{\the\@tempcnta}%
   \def\thefigure{\if@numart\else\thechapter.\fi
   \@arabic\c@merk\alph{figure}}%
   \let\resetsubfig=\r@setsubfig}

\def\samenumber{\addtocounter{\@captype}{-1}%
\@ifundefined{last@subfig}{}{\setcounter{merk}{\last@subfig}}}

% redefinition of the "bibliography" environment
%
\def\biblstarthook#1{\gdef\biblst@rthook{#1}}
%
\AtBeginDocument{%
\ifx\secbibl\undefined
   \def\bibsection{\chapter*{\refname}\markboth{\refname}{\refname}%
      \addcontentsline{toc}{chapter}{\refname}%
      \csname biblst@rthook\endcsname}
\else
   \def\bibsection{\section*{\refname}\markright{\refname}%
      \addcontentsline{toc}{section}{\refname}%
      \csname biblst@rthook\endcsname}
\fi}
\ifx\oribibl\undefined % Springer way of life
   \renewenvironment{thebibliography}[1]{\bibsection
         \global\let\biblst@rthook=\undefined
         \def\@biblabel##1{##1.}
         \small
         \list{\@biblabel{\@arabic\c@enumiv}}%
              {\settowidth\labelwidth{\@biblabel{#1}}%
               \leftmargin\labelwidth
               \advance\leftmargin\labelsep
               \if@openbib
                 \advance\leftmargin\bibindent
                 \itemindent -\bibindent
                 \listparindent \itemindent
                 \parsep \z@
               \fi
               \usecounter{enumiv}%
               \let\p@enumiv\@empty
               \renewcommand\theenumiv{\@arabic\c@enumiv}}%
         \if@openbib
           \renewcommand\newblock{\par}%
         \else
           \renewcommand\newblock{\hskip .11em \@plus.33em \@minus.07em}%
         \fi
         \sloppy\clubpenalty4000\widowpenalty4000%
         \sfcode`\.=\@m}
        {\def\@noitemerr
          {\@latex@warning{Empty `thebibliography' environment}}%
         \endlist}
   \def\@lbibitem[#1]#2{\item[{[#1]}\hfill]\if@filesw
        {\let\protect\noexpand\immediate
        \write\@auxout{\string\bibcite{#2}{#1}}}\fi\ignorespaces}
\else % original bibliography is required
   \let\bibname=\refname
   \renewenvironment{thebibliography}[1]
     {\chapter*{\bibname
        \@mkboth{\bibname}{\bibname}}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
\fi

\let\if@threecolind\iffalse
\def\threecolindex{\let\if@threecolind\iftrue}
\def\indexstarthook#1{\gdef\indexst@rthook{#1}}
\renewenvironment{theindex}
               {\if@twocolumn
                  \@restonecolfalse
                \else
                  \@restonecoltrue
                \fi
                \columnseprule \z@
                \columnsep 1cc
                \@nobreaktrue
                \if@threecolind
                   \begin{multicols}{3}[\chapter*{\indexname}%
                \else
                   \begin{multicols}{2}[\chapter*{\indexname}%
                \fi
                {\csname indexst@rthook\endcsname}]%
                \global\let\indexst@rthook=\undefined
                \markboth{\indexname}{\indexname}%
                \addcontentsline{toc}{chapter}{\indexname}%
                \flushbottom
                \parindent\z@
                \rightskip\z@ \@plus 40\p@
                \parskip\z@ \@plus .3\p@\relax
                \flushbottom
                \let\item\@idxitem
                \def\,{\relax\ifmmode\mskip\thinmuskip
                             \else\hskip0.2em\ignorespaces\fi}%
                \normalfont\small}
               {\end{multicols}
                \global\let\if@threecolind\iffalse
                \if@restonecol\onecolumn\else\clearpage\fi}

\def\idxquad{\hskip 10\p@}% space that divides entry from number

\def\@idxitem{\par\setbox0=\hbox{--\,--\,--\enspace}%
                  \hangindent\wd0\relax}

\def\subitem{\par\noindent\setbox0=\hbox{--\enspace}% second order
                \kern\wd0\setbox0=\hbox{--\,--\,--\enspace}%
                \hangindent\wd0\relax}% indexentry

\def\subsubitem{\par\noindent\setbox0=\hbox{--\,--\enspace}% third order
                \kern\wd0\setbox0=\hbox{--\,--\,--\enspace}%
                \hangindent\wd0\relax}% indexentry

\def\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}

\def\subtitle#1{\gdef\@subtitle{#1}}
\def\@subtitle{}

\def\maketitle{\par
 \begingroup
   \def\thefootnote{\fnsymbol{footnote}}%
   \def\@makefnmark{\hbox
       to\z@{$\m@th^{\@thefnmark}$\hss}}%
   \if@twocolumn
     \twocolumn[\@maketitle]%
     \else 
     \global\@topnum\z@   % Prevents figures from going at top of page.
     \@maketitle \fi\thispagestyle{empty}\@thanks
     \par\penalty -\@M
 \endgroup
 \setcounter{footnote}{0}%
 \let\maketitle\relax
 \let\@maketitle\relax
 \gdef\@thanks{}\gdef\@author{}\gdef\@title{}\let\thanks\relax}

\def\@maketitle{
 \null
 \vskip 2em                 % Vertical space above title.
\begingroup
  \def\and{\unskip, }
  \parindent=\z@
  \pretolerance=10000
  \rightskip=\z@ \@plus 3cm
  {\LARGE                   % each author set in \LARGE
   \lineskip .5em
   \@author
   \par}%
  \vskip 2cm                % Vertical space after author.
  {\Huge \@title \par}%     % Title set in \Huge size.
  \vskip 1cm                % Vertical space after title.
  \if!\@subtitle!\else
   {\LARGE\ignorespaces\@subtitle \par}
   \vskip 1cm                % Vertical space after subtitle.
  \fi
  % \if!\@date!\else
  %   {\large \@date}%          % Date set in \large size.
  %   \par
  %   \vskip 1.5em               % Vertical space after date.
  % \fi
 \vfill
 % {\Large Springer\par}
%\vskip 5\p@
%\large
%  Berlin\enspace Heidelberg\enspace New\kern0.1em York\\
%  Hong\thinspace Kong\enspace London\\
%  Milan\enspace Paris\enspace Tokyo\par
\endgroup}

% Useful environments
\newenvironment{acknowledgement}{\par\addvspace{17\p@}\small\rm
\trivlist\item[\hskip\labelsep{\it\ackname}]}
{\endtrivlist\addvspace{6\p@}}
%
\newenvironment{noteadd}{\par\addvspace{17\p@}\small\rm
\trivlist\item[\hskip\labelsep{\it\noteaddname}]}
{\endtrivlist\addvspace{6\p@}}
%
\renewenvironment{abstract}{%
      \advance\topsep by0.35cm\relax\small
      \labelwidth=\z@
      \listparindent=\z@
      \itemindent\listparindent
              \trivlist\item[\hskip\labelsep\bfseries\abstractname]%
              \if!\abstractname!\hskip-\labelsep\fi
      }
    {\endtrivlist}

% define the running headings of a twoside text
\def\runheadsize{\small}
\def\runheadstyle{\rmfamily\upshape}
\def\customizhead{\hspace{\headlineindent}}

\def\ps@headings{\let\@mkboth\markboth
   \let\@oddfoot\@empty\let\@evenfoot\@empty
   \def\@evenhead{\runheadsize\runheadstyle\rlap{\thepage}\customizhead
                  \leftmark\hfil}
   \def\@oddhead{\runheadsize\runheadstyle\hfil\rightmark\customizhead
                  \llap{\thepage}}
   \def\chaptermark##1{\markboth{{\ifnum\c@secnumdepth>\m@ne
      \thechapter\thechapterend\hskip\betweenumberspace\fi ##1}}{{\ifnum %!!!
      \c@secnumdepth>\m@ne\thechapter\thechapterend\hskip\betweenumberspace\fi ##1}}}%!!!
   \def\sectionmark##1{\markright{{\ifnum\c@secnumdepth>\z@
      \thesection\seccounterend\hskip\betweenumberspace\fi ##1}}}}

\def\ps@myheadings{\let\@mkboth\@gobbletwo
   \let\@oddfoot\@empty\let\@evenfoot\@empty
   \def\@evenhead{\runheadsize\runheadstyle\rlap{\thepage}\customizhead
                  \leftmark\hfil}
   \def\@oddhead{\runheadsize\runheadstyle\hfil\rightmark\customizhead
                  \llap{\thepage}}
   \let\chaptermark\@gobble
   \let\sectionmark\@gobble
   \let\subsectionmark\@gobble}


\ps@headings

\endinput
%end of file svmono.cls
