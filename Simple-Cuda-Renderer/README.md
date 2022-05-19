# Assignment 3: A Simple CUDA Renderer #
[Stanford CS149 Assignment3](https://github.com/stanford-cs149/asst3)


![My Image](handout/teaser.jpg?raw=true)

## Overview ##

In this assignment you will write a parallel renderer in CUDA that draws colored circles. 
While this renderer is very simple, parallelizing the renderer will require you to design and implement data structures 
that can be efficiently constructed and manipulated in parallel. This is a challenging
assignment so you are advised to start early. __Seriously, you are advised to start early.__ Good luck!

<pre>
------------
Score table: 
------------
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.6107           | 0.6049          | 12              |
| rand10k         | 13.2368          | 12.615          | 12              |
| rand100k        | 119.3456         | 124.5882        | 12              |
| pattern         | 1.2442           | 0.8663          | 12              |
| snowsingle      | 62.7321          | 23.1939         | 12              |
| biglittle       | 77.9342          | 85.9024         | 12              |
--------------------------------------------------------------------------
|                                    | Total score:    | 72/72           |
--------------------------------------------------------------------------
</pre>