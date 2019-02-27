#lang racket/base
(require pollen/private/version)
(provide (all-defined-out))

(module setup racket/base
  (provide (all-defined-out))
  (define project-server-port 8081))

(define (speaker-notes . text)
  `(aside ((class "notes")) ,@text))

(define (clojure . exprs)
  `(pre (code ((class "clojure")) ,@exprs)))
