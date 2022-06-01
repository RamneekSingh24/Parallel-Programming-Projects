#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>
#include <vector>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//

static double correction = 0.0;
static double global_diff = 0.0;
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  double *old_score = solution;
  double *new_score = new double[numNodes];
  std::vector<std::vector<int>> incoming(numNodes);

#pragma omp parallel for
  for (int i = 0; i < numNodes; ++i)
  {
    solution[i] = equal_prob;
    new_score[i] = 0.0;
  }

  double *del_ptr = new_score;

  std::vector<int> zero_out_degree;

#pragma omp parallel for
  for (int i = 0; i < numNodes; i++)
  {
    if (outgoing_size(g, i) == 0)
    {
#pragma omp critical
      zero_out_degree.emplace_back(i);
    }
  }

  bool converged = false;
  int itr = 0;
  while (!converged)
  {
    ++itr;
#pragma omp parallel for schedule(static, 512)
    for (int i = 0; i < numNodes; i++)
    {
      double add = old_score[i] / outgoing_size(g, i);

      const Vertex *start = outgoing_begin(g, i);
      const Vertex *end = outgoing_end(g, i);

      for (const Vertex *v = start; v != end; v++)
      {
        int j = *v;
        #pragma omp atomic update
        new_score[j] += add;
      }
    }

#pragma omp parallel for schedule(static, 512)
    for (int i = 0; i < numNodes; i++)
    {
      new_score[i] = (damping * new_score[i]) + (1.0 - damping) / numNodes;
    }

    int sz = zero_out_degree.size();
    correction = 0.0;
#pragma omp parallel for schedule(static, 512) reduction(+ \
                                           : correction)
    for (int i = 0; i < sz; i++)
    {
      correction += old_score[zero_out_degree[i]] * damping / numNodes;
    }

    global_diff = 0.0;
#pragma omp parallel for schedule(static, 512) reduction(+ \
                                           : global_diff)
    for (int i = 0; i < numNodes; i++)
    {
      new_score[i] += correction;
      global_diff += abs(new_score[i] - old_score[i]);
      old_score[i] = 0;
    }

    std::swap(old_score, new_score);

    converged = global_diff < convergence;
  }

  if (old_score != solution)
  {
    memcpy(solution, old_score, sizeof(double) * numNodes);
  }

  delete[] del_ptr;
}
