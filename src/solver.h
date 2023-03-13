// Create a solver class

#ifndef SOLVER_H
#define SOLVER_H

#include <vector>

class Solver
{
public:
    Solver();
    void Solve();
private:
    std::vector<double> m_x;
    std::vector<double> m_y;
};

