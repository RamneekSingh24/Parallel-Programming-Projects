# Assignment 4: Big Graph Processing in OpenMP #

**Due: Monday Nov 29th, 11:59PM PST**

**100 points total**

## Overview ##

In this assignment, you will implement two graph processing algorithms: [breadth-first search](https://en.wikipedia.org/wiki/Breadth-first_search) (BFS) and a simple implementation of [page rank](https://en.wikipedia.org/wiki/PageRank). A good implementation of this assignment will be able to run these algorithms on graphs containing hundreds of millions of edges on a multi-core machine in only seconds.

<pre>

Page rank
-------------------------------------------
SCORES :
-------------------------------------------
soc-livejournal1_68m.graph  |   4.00000 / 4 |
-------------------------------------------
com-orkut_117m.graph        |   4.00000 / 4 |
-------------------------------------------
rmat_200m.graph             |   4.00000 / 4 |
-------------------------------------------
random_500m.graph           |   4.00000 / 4 |
-------------------------------------------
TOTAL                       |   16.00000 / 16 |


Hybrid breadth first search  
--------------------------------------------------------------------------
SCORES :                    |   Top-Down    |   Bott-Up    |    Hybrid    |
--------------------------------------------------------------------------
grid1000x1000.graph         |      2.00 / 2 |     3.00 / 3 |     3.00 / 3 |
--------------------------------------------------------------------------
soc-livejournal1_68m.graph  |      2.00 / 2 |     1.39 / 3 |     3.00 / 3 |
--------------------------------------------------------------------------
com-orkut_117m.graph        |      2.00 / 2 |     1.51 / 3 |     3.00 / 3 |
--------------------------------------------------------------------------
random_500m.graph           |      7.00 / 7 |     8.00 / 8 |     8.00 / 8 |
--------------------------------------------------------------------------
rmat_200m.graph             |      7.00 / 7 |     8.00 / 8 |     8.00 / 8 |
--------------------------------------------------------------------------
TOTAL                                                      |  66.90 / 70 |
</pre>