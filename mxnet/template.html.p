<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <meta name="author" content="(Dave Liepmann)"/>
    
    <title>MXNet @ ClojureD 2019</title>
    <link rel="stylesheet" href="./reveal.js-3.6.0/css/reveal.css"/>
    <link rel="stylesheet" href="./reveal.js-3.6.0/lib/css/zenburn.css"/>

    <link rel="stylesheet" href="./reveal.js-3.6.0/css/theme/night.css" id="theme"/>
    <link rel="stylesheet" href="./reveal.js-3.6.0/plugin/highlight/highlight.js" id="highlight"/>

    <link rel="stylesheet" href="./talk-mxnet.css"/>
  </head>
  <body>
    <div class="reveal">
      <div class="slides">
        ◊(->html doc #:splice? #t)
      </div>
    </div>
    <script src="./reveal.js-3.6.0/lib/js/head.min.js"></script>

    <script src="./reveal.js-3.6.0/js/reveal.js"></script>

    <script>
      // Full list of configuration options available here:
      // https://github.com/hakimel/reveal.js#configuration
      Reveal.initialize({

      controls: true,
      progress: true,
      history: false,
      center: true,
      slideNumber: false,
      rollingLinks: false,
      keyboard: true,
      overview: true,

      theme: Reveal.getQueryHash().theme, // available themes are in /css/theme
      transition: Reveal.getQueryHash().transition || 'default', // default/cube/page/concave/zoom/linear/fade/none
      transitionSpeed: 'default',
      multiplex: {
      secret: '', // null if client
      id: '', // id, obtained from socket.io server
      url: '' // Location of socket.io server
      },

      // Optional libraries used to extend on reveal.js
      dependencies: [
      { src: './reveal.js-3.6.0/lib/js/classList.js', condition: function() { return !document.body.classList; } },
      { src: './reveal.js-3.6.0/plugin/markdown/marked.js', condition: function() { return !!document.querySelector( '[data-markdown]' ); } },
      { src: './reveal.js-3.6.0/plugin/markdown/markdown.js', condition: function() { return !!document.querySelector( '[data-markdown]' ); } },
      { src: './reveal.js-3.6.0/plugin/zoom-js/zoom.js', async: true, condition: function() { return !!document.body.classList; } },
      { src: './reveal.js-3.6.0/plugin/notes/notes.js', async: true, condition: function() { return !!document.body.classList; } },
      { src: './reveal.js-3.6.0/plugin/highlight/highlight.js', async: true, callback: function() { hljs.initHighlightingOnLoad(); } }]
      });
    </script>
  </body>
</html>
