#lang pollen

◊section{
    ◊h3{tricking sand into ◊span[#:class "dal--tasks"]{thinking}}
    ◊h4{◊span[#:class "dal--highlight"]{deep learning} in clojure with ◊span[#:class "dal--highlight"]{apache mxnet}}
    ◊a[#:target "_blank" #:href "https://twitter.com/daisyowl/status/841802094361235456?lang=en"]{◊img[#:width "80%" #:src "images/tweet1.png"]{}}
    ◊h4[#:style "margin-bottom: 0px;"]{
        ◊span[#:style "vertical-align: middle; margin-right: 0.3em;"]{@daveliepmann}}

        ◊; Applied logo is 111111 because it is modified from orthodox background color of 181818
        ◊a[#:href "http://www.appliedscience.studio/"]{◊img[#:src "images/applied-logomatic-111111.svg" #:class "dal--no-border" #:style "vertical-align: middle; display: inline;" #:width "16%"]}
    ◊speaker-notes{
        ◊p{I'm here to talk about deep learning in Clojure with Apache MXNet.}
        ◊p{MXNet is OSS DL framework for Python, Julia, Scala, aaaaaand now Clojure}
        ◊p{Goal is for you to see what problems deep learning and MXNet can solve, and to inspire you to go try it out.}
        ◊p{NEXT: To get everyone on the same page... }}}


◊;;;;;;;;;; PROBLEM STATEMENT
◊;;;;;;;;;; PROBLEM STATEMENT
◊;;;;;;;;;; PROBLEM STATEMENT
◊;;;;;;;;;; PROBLEM STATEMENT
◊;;;;;;;;;; PROBLEM STATEMENT
◊;;;;;;;;;; PROBLEM STATEMENT
◊;;;;;;;;;; PROBLEM STATEMENT
◊;;;;;;;;;; PROBLEM STATEMENT
◊;;;;;;;;;; PROBLEM STATEMENT
◊;;;;;;;;;; PROBLEM STATEMENT

◊;;;; Seeing/MNIST
◊section{
    ◊h2[#:class "dal--fill-line dal--fill-line-title dal--tricking"]{TRICKING SAND INTO ◊span[#:class "dal--tasks"]{READING}}
    ◊speaker-notes{
        ◊p{let's introduce these terms ("DL", "NNs") with an example of computer vision}}}

◊section{
    ◊a[#:href "https://ml4a.github.io/ml4a/neural_networks/"]{◊img[#:src "images/fig_mnist_groundtruth.png"]{}}
    ◊speaker-notes{
        ◊ul{
            ◊li{Take a classic example, the MNIST handwritten digit database.}
            ◊li{The goal is for the computer to identify which number 0-9 is written.}}}}

◊section{
    ◊a[#:href "https://ml4a.github.io/ml4a/looking_inside_neural_nets/"]{◊img[#:src "images/mnist-input.png" #:style "background: white;"]{}}
    ◊speaker-notes{
        ◊ul{
            ◊li{You can do that by encoding the image pixel-by-pixel}
            ◊li{...then rolling out that square of pixels into a line that makes up the input layer of our 1-layer neural network}
            ◊li{Those map directly to the output side: one neuron per possible outcome}}}}

◊section{
    ◊a[#:href "https://ml4a.github.io/demos/forward_pass_mnist/"]{◊img[#:src "images/mnist-forward-pass.png" #:style "background: white;"]{}}
    ◊speaker-notes{
        ◊ul{
            ◊li{each output layer represents probability of a particular digit}
            ◊li{it's pretty confident it sees a 5}
            ◊li{"deep" comes in w/added layer in the middle}
            ◊li{◊strong{Deep learning work = building and training networks like these}}}}}

◊section{
    ◊img[#:class "dal--mxnet-logo dal--no-border" #:style "width: 30%; float: left;" #:src "images/mxnet_logo.png"]
    ◊span[#:class "dal--slogan dal--mnist"]{MNIST network ◊span[#:class "dal--highlight"]{in Clojure}}
    ◊pre{
        ◊code[#:class "clojure small"]{
(defn get-symbol []
  (as-> (sym/variable "data") data
    (sym/fully-connected "fc1"    {:data data :num-hidden 128})
    (sym/activation "relu1"       {:data data :act-type "relu"})
    (sym/fully-connected "fc2"    {:data data :num-hidden 64})
    (sym/activation "relu2"       {:data data :act-type "relu"})
    (sym/fully-connected "fc3"    {:data data :num-hidden 10})
    (sym/softmax-output "softmax" {:data data})))}}    
    ◊speaker-notes{
        ◊ul{
            ◊li{There is more glue code involved, and you have to train the network for a while}
            ◊li{defining the network is the heart of implementation}
            ◊li{NEXT: another example, for more complex architectures}}}}


◊;;;; Poetry/LSTM
◊section{
    ◊h2[#:class "dal--fill-line dal--fill-line-title dal--tricking"]{TRICKING SAND INTO ◊span[#:class "dal--tasks"]{WRITING POETRY}}
    ◊speaker-notes{
        ◊ul{
            ◊li{Sometimes the network or its nodes get quite complex}
            ◊li{NEXT: For instance, if you were implementing this paper that introduced LSTMs...}}}}

◊section{
    ◊img[#:src "images/lstm-hochreiter-schmidhuber.png"]{}
    ◊speaker-notes{
        ◊ul{
            ◊li{a kind of RNN}}}}

◊section{
    ◊img[#:src "images/lstm-cell.png"]{}
    ◊speaker-notes{
        ◊p{Much more complicated internal structure than the nodes we used to solve MNIST}
        ◊p{NEXT: MXNet for Clojure ships with example code for this...}}}

◊section{
        ◊img[#:class "dal--mxnet-logo dal--no-border" #:style "width: 30%; float: left;" #:src "images/mxnet_logo.png"]
        ◊span[#:class "dal--slogan dal--lstm"]{LSTM: from paper ◊span[#:class "dal--highlight"]{to Clojure}}
        ◊pre{
            ◊code[#:class "clojure small"]{
    (defn lstm [num-hidden in-data prev-state param seq-idx layer-idx dropout]
      (let [i2h (sym/fully-connected (str "t" seq-idx "_l" layer-idx "_i2h")
                                     {:data in-data :weight (:i2h-weight param)
                                      :bias (:i2h-bias param) :num-hidden (* num-hidden 4)})
            h2h (sym/fully-connected (str "t" seq-idx "_l" layer-idx "_h2h")
                                     {:data (:h prev-state) :weight (:h2h-weight param)
                                      :bias (:h2h-bias param) :num-hidden (* num-hidden 4)})
            gates (sym/+ i2h h2h)
            slice-gates (sym/slice-channel (str "t" seq-idx "_l" layer-idx "_slice") {:data gates :num-outputs 4})
            in-gate      (sym/activation {:data (sym/get slice-gates 0) :act-type "sigmoid"})
            in-transform (sym/activation {:data (sym/get slice-gates 1) :act-type "tanh"})
            forget-gate  (sym/activation {:data (sym/get slice-gates 2) :act-type "sigmoid"})
            out-gate     (sym/activation {:data (sym/get slice-gates 3) :act-type "sigmoid"})]
        (lstm-state (sym/+ (sym/* forget-gate (:c prev-state))
                           (sym/* in-gate in-transform))
                    (sym/* out-gate (sym/activation {:data next-c :act-type "tanh"})))))
    }}
        ◊speaker-notes{
            ◊p{Just by asking the network "what letter is probably next?" it eventually figures out English words (sort of)}
            ◊p{NEXT: if you train a network of those nodes on 1MB corpus of Obama speeches...}}}


◊;;;;;;;;;; EXTRA EXAMPLES
◊;;;;;;;;;; EXTRA EXAMPLES
◊;;;;;;;;;; EXTRA EXAMPLES
◊;;;;;;;;;; EXTRA EXAMPLES
◊;;;;;;;;;; EXTRA EXAMPLES
◊;;;;;;;;;; EXTRA EXAMPLES
◊;;;;;;;;;; EXTRA EXAMPLES
◊;;;;;;;;;; EXTRA EXAMPLES
◊;;;;;;;;;; EXTRA EXAMPLES
◊;;;;;;;;;; EXTRA EXAMPLES

◊section{    
    ◊section{
        ◊p[#:class "dal--console"]{◊span[#:class "dal--highlight"]{The joke}  o Iso nt   thoo  ief s  o se lds  ,   por rs e  maa tyoir at oro slk i lely  eerre   Whoethaaliis e  tthoo o actitoou msea  to utsu ,  s t ratthhee oainrgielnearip er  pte e r  da  int htahoe}
        ◊p[#:class "dal--console fragment"]{◊span[#:class "dal--highlight"]{The joke} schools to open health care and every child or whether or children at a single party that makes America adved-us to callying as new technology to early halfalishs of the wares.TF}
        ◊speaker-notes{
            ◊p{...you get output that initially looks like this.}
            ◊p{(Highlighted part is the human-written starter.)}
            ◊p{Just by asking the network "what letter is probably next?" it eventually figures out English words (sort of)}
            ◊p{NEXT: this is great, but...}}}

    ◊section{
        ◊h2[#:class "dal--fill-line dal--fill-line-title dal--tricking"]{TRICKING SAND INTO ◊span[#:class "dal--tasks"]{MORE EXAMPLES}}}

    ◊;;;; Object detection
    ◊section{
        ◊h2[#:class "dal--fill-line dal--fill-line-title dal--tricking"]{TRICKING SAND INTO ◊span[#:class "dal--tasks"]{SEEING}}
        ◊speaker-notes{
            ◊p{Let's look at another example that ships (in Clojure) with MXNet}}}
    
    ◊section{
        ◊p[#:class "dal--attribution"]{◊a[#:href "https://arxiv.org/pdf/1512.02325.pdf"]{
            ◊img[#:src "images/ssmd.png"]{}}}
        ◊speaker-notes{
            ◊p{Another paper implementation--this is the coolest aspect of MXNet: that you can explore prebuilt implementations out of the box, and you have the tools to modify them or write your own ◊em{immediately}.}
            ◊p{that it's a framework is exciting b/c you get a leg up on building real deep learning tools}}}
    
    ◊section{
        ◊div{
            ◊img[#:width "34%" #:src "images/mxnet-object-detection.jpg"]{}
            ◊img[#:width "61%" #:src "images/mxnet-object-detection2.jpg"]{}}
        ◊p[#:class "dal--attribution"]{(thanks to contributors Kedar Bellare and Nicolas Modrzyk)}
        ◊p[#:class "dal--attribution"]{◊a[#:href "http://gigasquidsoftware.com/blog/2019/01/19/object-detection-with-clojure-mxnet/"]{Carin Meier blog post}}
        ◊speaker-notes{
            This example is so fresh it requires a nightly build of MXNet (b/c it needs `infer` package)}}

    
    ◊;;;; Painting/Neural Style
    ◊section{
        ◊h2[#:class "dal--fill-line dal--fill-line-title dal--tricking"]{TRICKING SAND INTO ◊span[#:class "dal--tasks"]{PAINTING}}
        ◊speaker-notes{}}
    
    ◊section{
        ◊img[#:src "images/style-transfer-paper.png" #:class "dal--no-border"]{}
        ◊speaker-notes{
            2015 paper out of Tubingen
            https://arxiv.org/pdf/1508.06576.pdf
            (NOT THE SAME as GANs that Carin talks about--watch her talk for that)}}

    ◊section{
        ◊img[#:src "images/style-transfer-tubingen.png"
             #:class "dal--no-border"
             #:width "60%"]
        ◊speaker-notes{
            ◊ul{
                ◊li{A. Starry Night, VAN GOGH}
                ◊li{B. The Shipwreck of the Minotaur, J.M.W. Turner, 1805}
                ◊li{C. Starry Night, van Gogh, 1889}
                ◊li{D. Der Schrei, Edvard Munch,1893}
                ◊li{E. Femme nue assise, Pablo Picasso, 1910}
                ◊li{F. Composition VII, Wassily Kandinsky,1913}}}}
    
    ◊section{
        ◊h3{Neural Style Transfer}
        ◊img[#:src "images/style-transfer-fig1.png" #:width "80%"]{}
        ◊speaker-notes{
            given an input image and a style image, what would the input image look like if it were "drawn like" the style?
            uses convolutional DNNs to extract _relationships_ between pixels over space}}
    
    ◊section{
        ◊div{
            ◊img[#:width "49%" #:src "images/neural/input.png"]{}
            ◊img[#:width "49%" #:src "images/neural/style.png"]{}}
        ◊speaker-notes{
            Clojure version of MXNet provides example code for this OUT OF THE BOX
            input: "starry night" and ???
            NEXT: because Carin implemented the paper in Clojure,
            you can calculate the left image "in the style of" starry night
            (with Clojure!)}}
    
    ◊section{
        ◊img[#:width "30%" #:src "images/neural/out_0.png"]{}
        ◊img[#:width "30%" #:src "images/neural/out_16.png"]{}
        ◊img[#:width "30%" #:src "images/neural/out_22.png"]{}
        ◊img[#:src "images/neural/final.png"]{}}}


◊;;;;;;;;;; INDUSTRY OVERVIEW
◊;;;;;;;;;; INDUSTRY OVERVIEW
◊;;;;;;;;;; INDUSTRY OVERVIEW
◊;;;;;;;;;; INDUSTRY OVERVIEW
◊;;;;;;;;;; INDUSTRY OVERVIEW
◊;;;;;;;;;; INDUSTRY OVERVIEW
◊;;;;;;;;;; INDUSTRY OVERVIEW
◊;;;;;;;;;; INDUSTRY OVERVIEW
◊;;;;;;;;;; INDUSTRY OVERVIEW
◊;;;;;;;;;; INDUSTRY OVERVIEW

◊section{
    ◊h2[#:class "dal--fill-line dal--fill-line-title dal--tricking"]{TRICKING SAND ◊br{}◊span[#:class "dal--tasks"]{WITH CLOJURE}}
    ◊speaker-notes{
        ◊p{How does this all work?}
        ◊p{...and why did it take so long to get this in Clojure?}
        ◊p{NEXT: who is the king of the DL jungle?}}}

◊section{
    ◊img[#:src "images/green-tree-python.jpg" #:width "100%"]{}
    ◊speaker-notes{
        ◊p{Python is the king of the DL jungle, and proglangs like Julia & R pick up the scraps.}
        ◊p{NEXT: take a step back}}}

◊section{
    ◊a[#:href "https://ml4a.github.io/demos/forward_pass_mnist/"]{◊img[#:src "images/karpathy-weights.png" #:style "background: white;"]{}}
    ◊speaker-notes{
        ◊p{DL applications = building large networks}
        ◊p{NNs require many, many calculations between all the nodes and the weights of the edges between them}
        ◊p{◊strong{NEXT}: How many calculations? LOTS...}}}

◊section{
    ◊br{}
    ◊p{"Large research groups [have the resources] to tune models on ◊strong{450 GPUs for 7 days}"}
    ◊br{}
    ◊img[#:src "images/winners-curse-title.png"]{}    
    ◊speaker-notes{
        ◊p{In many software domains, performance isn't a concern, or isn't a big deal until scale.}
        ◊p{In DL perf is ◊strong{always} a big deal.}
        ◊p{NEXT: how do langs like Python & Julia get performance for that scale?}}}

◊section{
    ◊a[#:href "https://medium.com/@erikhallstrm/backpropagation-from-the-beginning-77356edf427d"]{◊img[#:src "images/nn-matrix-op.png" #:style "background: white; width: 55%;"]{}}
    ◊speaker-notes{
        ◊p{best way to perform these ops is as matrices}
        ◊p{◊strong{NEXT}: so in DL we need a way to run matrix operations really fast}}}

◊;;;;;;;;;; BLAS & LAPACK
◊;;;;;;;;;; BLAS & LAPACK
◊;;;;;;;;;; BLAS & LAPACK
◊;;;;;;;;;; BLAS & LAPACK
◊;;;;;;;;;; BLAS & LAPACK
◊;;;;;;;;;; BLAS & LAPACK
◊;;;;;;;;;; BLAS & LAPACK
◊;;;;;;;;;; BLAS & LAPACK
◊;;;;;;;;;; BLAS & LAPACK
◊;;;;;;;;;; BLAS & LAPACK

◊section{
    ◊h2[#:class "dal--fill-line dal--fill-line-title dal--tricking"]{TRICKING SAND INTO}
    ◊h2[#:class "dal--fill-line dal--fill-line-title dal--tricking"]{◊span[#:class "dal--tasks"]{FAST LINEAR ALGEBRA}}
    ◊speaker-notes{
        ◊p{enter: linear algebra (domain of maths dealing with matrices)}}}

◊section{
    ◊img[#:src "images/Matrix_multiplication_qtl1.svg" #:style "background: white;" #:width "45%"]{}
    ◊img[#:src "images/GPU_NVIDIA_NV45_ES_GPU.jpg" #:width "45%"]{}
    ◊speaker-notes{
        ◊p{this has been a program in graphics programming for a while}
        ◊p{so GPUs are optimized (via parallelism) to rapidly handle computation if arranged as large-scale linear algebra}
        ◊p{◊strong{NEXT}: but to get this parallelism-driven speedup, you need to be able to talk to the hardware...}}}

◊section{
    ◊h4{BLAS: Basic Linear Algebra Subprograms}
    ◊a[#:href "http://www.netlib.org/blas/blasqr.pdf"]{◊img[#:src "images/blas.png"]{}}
    ◊speaker-notes{
        ◊p{Fast matrix ops means BLAS, an open standard for low-level linear algebra number crunching}
        ◊p{implementations of this standard are C and FORTRAN modules with ◊em{decades} of hand-tuning}
        ◊p{◊strong{NEXT}: to talk to BLAS, you'll also want another low-level lib called...}
        ◊; ◊p{routines for vector addition, scalar multiplication, dot products, linear combinations, and matrix multiplication}
        ◊; "BLAS is the low–level part of your system that is responsible for efficiently performing numerical linear algebra, i.e. all the heavy number crunching."
        ◊; - http://markus-beuckelmann.de/blog/boosting-numpy-blas.html
        }}

◊section{
    ◊h4{LAPACK: Linear Algebra PACKage}
    ◊img[#:src "images/LAPACK_logo.svg" #:width "60%" #:style "background: white;"]{}
    ◊speaker-notes{
        ◊p{LAPACK: implements BLAS, one level of abstraction up.}
        ◊p{also extremely fast because of decades of optimization}
        ◊; (routines for solving systems of linear equations and linear least squares, eigenvalue problems, and singular value decomposition)
        ◊p{Success in deep learning = access to these C/FORTRAN libs.}}}

◊section{
    ◊h2{Access to BLAS/LAPACK}
    ◊ul{
        ◊li{Python: NumPy (1995)}
        ◊li{Python: SciPy (2001)}
        ◊li{Julia: since birth (2012)}}
    ◊speaker-notes{
        ◊p{Python has had access to C/FORTRAN since forever.}
        ◊p{Julia purpose-built for it.}
        ◊p{◊strong{NEXT}: in contrast, JVM access came late}}}
        
◊section{
    ◊h2{JVM access to BLAS/LAPACK}
    ◊ul{
        ◊li{interop: f2j}
        ◊li{Neanderthal}
        ◊li{Clatrix}
        ◊li{MXNet}
        ◊li{Deep Learning for Java / DL4CLJ}}
    ◊speaker-notes{
        ◊p{first 3 are direct access, useful but not simple or performant to get started with DL}
        ◊p{final 2 are frameworks, providing API & tools on top}
        ◊p{◊strong{NEXT}: let's look at overall architecture}}}


◊;;;;;;;;;; MXNet overview
◊;;;;;;;;;; MXNet overview
◊;;;;;;;;;; MXNet overview
◊;;;;;;;;;; MXNet overview
◊;;;;;;;;;; MXNet overview
◊;;;;;;;;;; MXNet overview
◊;;;;;;;;;; MXNet overview
◊;;;;;;;;;; MXNet overview
◊;;;;;;;;;; MXNet overview
◊;;;;;;;;;; MXNet overview

◊section[#:data-transition "convex-in none-out"]{
    ◊img[#:class "dal--mxnet-logo dal--no-border" #:style "width: 30%; float: left;" #:src "images/mxnet_logo.png"]
    ◊span[#:class "dal--slogan dal--mxnet-slogan dal--highlight"]{"Modern open-source deep learning framework"}
    ◊div[#:style "clear: both;"]{}
    ◊div{
        ◊img[#:class "dal--no-border dal--no-bg" #:src "images/mxnet overview fig1.png"]{}}
    ◊speaker-notes{
        ◊p{bird's-eye view of MXNet}
        ◊ol{
            ◊li{"sand" at bottom}
            ◊li{BLAS/LAPACK layer to efficiently talk to hardware}
            ◊li{above that: MXnet abstraction layer}
            ◊li{high-level lang describes networks, how to train them}}
        ◊p{◊strong{NEXT}: how does Clojure fit into this?}}}

◊section[#:data-transition "none"]{
    ◊img[#:class "dal--mxnet-logo dal--no-border" #:style "width: 30%; float: left;" #:src "images/mxnet_logo.png"]
    ◊span[#:class "dal--slogan dal--mxnet-slogan dal--highlight"]{"Modern open-source deep learning framework"}
    ◊div[#:style "clear: both;"]{}
    ◊div{
        ◊img[#:class "dal--no-border dal--no-bg" #:src "images/mxnet overview with scala.png"]{}}
    ◊speaker-notes{
        ◊p{one supported lang is Scala, a "better Java", which compiles to JVM bytecode.}
        ◊p{That's fine for Scala folks, tho I don't particularly like Scala.}
        ◊p{◊strong{NEXT}: but it gives us a foothold!}}}

◊section[#:data-transition "none"]{
    ◊img[#:class "dal--mxnet-logo dal--no-border" #:style "width: 30%; float: left;" #:src "images/mxnet_logo.png"]
    ◊span[#:class "dal--slogan dal--mxnet-slogan dal--highlight"]{"Modern open-source deep learning framework"}
    ◊div[#:style "clear: both;"]{}
    ◊div{
        ◊img[#:class "dal--no-border dal--no-bg" #:src "images/mxnet overview with scala and clojure.png"]{}}
    ◊speaker-notes{
        ◊p{anywhere the JVM is, Clojure can go too.}
        ◊p{Carin Meier wrote Clojure bindings that piggieback on the Scala bindings, giving us native access to MXNet API}
        ◊p{◊strong{NEXT}: so now that all this is possible, what do we do?}}}

◊section{
    ◊h2{HELP WANTED}
    ◊ul{
        ◊li{use it!}
        ◊li{add examples}
        ◊li{debug/improve existing examples}
        ◊li{port new functionality}
        ◊li{write documentation & guides}}
    ◊speaker-notes{
        ◊p{Clojure is still a contrib project -- not a 1st class citizen}
        ◊p{we need support & users}
        ◊p{take it for a spin!}}}

◊section{
    ◊a[#:href "https://cwiki.apache.org/confluence/display/MXNET/Clojure+Package+Contribution+Needs"]{
              ◊img[#:src "images/clojure contribution screenshot.png"]{}}
    ◊speaker-notes{
        ◊p{We (mostly Carin) maintains a page of contribution needs.}
        ◊p{So: run an example, file bugs, contribute if you have time.}}}


◊;;;;;;;;;; DONE
◊;;;;;;;;;; DONE
◊;;;;;;;;;; DONE
◊;;;;;;;;;; DONE
◊;;;;;;;;;; DONE
◊;;;;;;;;;; DONE
◊;;;;;;;;;; DONE
◊;;;;;;;;;; DONE
◊;;;;;;;;;; DONE
◊;;;;;;;;;; DONE

◊section{
    ◊section{
        ◊h4[#:style "margin-top: 1em;"]{thanks!}
        ◊div[#:class "attribution"]{
            ◊ul{
                ◊li{Carin Meier & Apache MXNet contributors}
                ◊li{Racket & ◊a[#:href "https://docs.racket-lang.org/pollen/"]{Pollen} (these slides are a lisp program)}
                ◊li{◊a[#:href "https://openreview.net/pdf?id=rJWF0Fywf"]{Winner's Curse?} paper by Sculley, Snoek, Rahimi, Wiltschko}
                ◊li{◊a[#:href "https://ml4a.github.io/ml4a/"]{Machine Learning for Artists}, Gene Kogan (MNIST example)}
                ◊li{◊a[#:href "https://medium.com/@erikhallstrm/backpropagation-from-the-beginning-77356edf427d"]{Backpropagation from the beginning}, Erik Hallström}}}
        ◊h4{◊a[#:href "http://www.appliedscience.studio/"]{◊img[#:src "images/applied-logomatic-111111.svg" #:class "dal--no-border" #:style "vertical-align: middle; display: inline;" #:width "35%"]}}}

    ◊;;;;;;;;;; EXTRAS
    ◊;;;;;;;;;; EXTRAS
    ◊;;;;;;;;;; EXTRAS
    ◊;;;;;;;;;; EXTRAS
    ◊;;;;;;;;;; EXTRAS
    ◊;;;;;;;;;; EXTRAS - these are _intentionally_ within the final slide section, which is how we make them accessible by up/down arrows
    ◊;;;;;;;;;; EXTRAS
    ◊;;;;;;;;;; EXTRAS
    ◊;;;;;;;;;; EXTRAS
    ◊;;;;;;;;;; EXTRAS
    ◊;;;;;;;;;; EXTRAS
    
    ◊section{
        ◊h2[#:class "dal--fill-line dal--fill-line-title dal--tricking"]{TRICKING SAND INTO ◊span[#:class "dal--tasks"]{EXTRA SLIDES}}}
    
    ◊section{
        ◊div{
            ◊img[#:style "background-color: #111; float: left; margin-right: 2em;" #:width "35%" #:src "images/DL4J.png"]{}
            ◊span[#:class "dal--slogan dal--dl4j-slogan"]{Open source, distributed, deep learning library for the JVM}}
        ◊pre{
            ◊code[#:class "clojure small"]{
    (def model
      (-> (Word2Vec$Builder.)
          (.minWordFrequency 5)
          (.iterations 1)
          (.layerSize 100)
          (.seed 42)
          (.windowSize 5)
          (.iterate (BasicLineIterator. "resources/raw_sentences.txt"))
          (.tokenizerFactory (doto (DefaultTokenizerFactory.)
                               (.setTokenPreProcessor (CommonPreprocessor.))))
          (.build)))}}
        ◊br{}
        ◊pre{
            ◊code[#:class "clojure small"]{
    (.getWordVectorMatrix model "day")
    ;; #object[org.nd4j.linalg.cpu.nativecpu.NDArray 0x59073fb5 "[0.41,  0.21,  0.15,  -0.21,  -0.04,  -0.40,  -0.12,  -0.10,  -0.32,  0.35,  0.21,  0.28,  0.12,  -0.07,  0.05,  -0.07,  -0.20,  0.21,  0.14,  -0.15,  0.07,  0.20,  0.42,  -0.23,  0.10,  -0.40,  0.11,  -0.42,  -0.19,  -0.11,  0.29,  -0.00,  0.46,  -0.51,  0.14,  -0.23,  0.08,  -0.21,  -0.07,  0.10,  -0.31,  -0.19,  0.11,  0.21,  -0.07,  -0.12,  -0.47,  -0.16,  0.16,  -0.14,  0.28,  0.04,  0.24,  -0.14,  -0.35,  0.09,  -0.24,  -0.07,  0.16,  -0.46,  -0.28,  -0.01,  0.15,  0.43,  0.16,  0.04,  0.04,  0.19,  -0.25,  -0.35,  0.24,  -0.06,  0.18,  -0.01,  -0.03,  0.10,  0.06,  0.11,  0.13,  0.04,  -0.03,  -0.19,  -0.45,  0.12,  -0.00,  0.04,  0.17,  -0.34,  -0.03,  -0.18,  -0.11,  0.01,  0.15,  -0.06,  -0.19,  0.25,  0.01,  0.28,  -0.32,  -0.11]"]}}
        ◊speaker-notes{
            DL4J totally valid, just different flavor
            both frameworks are frustratingly filled with black box java objects & non-clojurey workflow
            I even have a Word2Vec blog post using DL4J waiting to publish
            "Java AI developers can use ND4J to define _N-_dimensional arrays in Java, which allows them to perform tensor operations on a JVM. ND4J uses “off-heap” memory outside the JVM to store tensors. The JVM only holds pointers to this external memory, and Java programs pass these pointers to the ND4J C++ back-end code through the Java Native Interface (JNI). This structure provides better performance while working with tensors from native code such as Basic Linear Algebra Subprograms (BLAS) and CUDA libraries."
            - https://developer.ibm.com/articles/cc-get-started-deeplearning4j/}}
    
    ◊section{
        ◊h3{MXNet vs DL4J?}
        ◊h1{◊span[#:class "dal--highlight"]{IS MIR EGAL.}}
        ◊img[#:src "images/ismiregal.gif"]{}}


    ◊section{
        ◊h3{Timeline}
        ◊ul{
            ◊li{1995 - Numpy (as Numeric)}
            ◊li{1999 - f2j (*)}
            ◊li{2001 - SciPy}
            ◊li{2006 - Numpy}
            ◊li{...}
            ◊li{2014 - DL4J}
            ◊li{2015 - TensorFlow, Keras, MXNet}}}

    ◊section{
        ◊h3{f2j performance}
        ◊a[#:href "http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.228.2895&rep=rep1&type=pdf"]{
            ◊img[#:src "images/f2j perf.png" #:width "60%"]}
        ◊p{◊em{Automatic translation of Fortran to JVM bytecode}, Keith Seymour and Jack Dongarra 2003}}}