# Talk: Deep Learning in Clojure with Apache MXNet 

Presented at [ClojureD](https://clojured.de/) 2019 by [Dave](https://twitter.com/daveliepmann/).

Made with [Pollen](https://docs.racket-lang.org/pollen/), so the presentation is a program written in lisp. If you're exploring, I recommend reading the Pollen docs first or in parallel. It is a great joy to write a talk as a Racket program that produces bog-standard HTML. Non-programmers and programmers alike should appreciate Pollen's top-notch, elegant documentation. Clojure programmers in particular can learn a lot from their lispy cousin Racket.


## Instructions
To compile the talk on your machine:

 1. install Pollen
 2. install [reveal.js](https://github.com/hakimel/reveal.js/#installation) to this directory
 3. start a pollen server in this directory
 4. go to http://localhost:8081/mxnet-clojured.html
 5. see live changes:
    - make some textual change to `mxnet-clojured.html.pm` (e.g. switch "sand" with "rock" inside the first `h3`)
    - reload the browser window
    - the Pollen server will automatically detect the change and produce fresh HTML \o/
    
The program is structured such that 99% of writing the talk happens in `mxnet-clojured.html.pm`. The `pollen.rkt` file manages the pollen server and provides a few helper functions (for instance, `◊speaker-notes`) written in lisp (Racket). The HTML-in-Pollen-DSL becomes slides through reveal.js, which is set up in `template.html.p` and the way that`mxnet-clojured.html.pm` is structured: each `◊section` is a slide. Please don't look at my mess of CSS.


## Workflow
I write in emacs using [Junsong Li's pollen-mode](https://github.com/lijunsong/pollen-mode) (NB: there is another pollen-mode and the two unfortunately share a name) with paredit turned on so I can select and edit structural units rather than text. I use the Pollen DSL, though Markdown and "lisp-style" Pollen can be used as well.

My emacs files include the following Pollen-specific config:

``` emacs-lisp
;; For Pollen in Racket. See https://docs.racket-lang.org/pollen/pollen-command-syntax.html
(global-set-key "\M-/" "◊")

(add-hook 'pollen-mode-hook
          (lambda ()
            (paredit-mode)
            (setq comment-start "◊;")))
```
    
Pollen-mode is quite useful but there are some places (such as indentation) where it could use some love, if you're in a mood to contribute to open source software.
