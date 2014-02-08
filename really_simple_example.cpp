/*
 * Copyright (c) 2011 Seiya Tokui <beam.web@gmail.com>
 * Copyright (c) 2014 Burkhard Ritter <burkhard@ualberta.ca>
 * This code is distributed under the MIT license.
 *
 * A very simple example program demonstrating how to use Arpaca.
 *
 * To compile this program: 
 * g++ \
 *    -I [/path/to/eigen] \
 *    -L [/path/to/libarpack.a] \
 *    -larpack \
 *    -o really_simple_example \
 *    really_simple_example.cpp 
 *
 * This assumes that arpaca.hpp is in the same directory.
 */

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "arpaca.hpp"
using namespace Eigen;
using namespace arpaca;

int main()
{
    // Matrix dimension
    const int n_dim = 4;
    // Number of eigenvalues to compute
    const int n_ev = 3;
    // Which eigenvalues to compute
    const EigenvalueType type = ALGEBRAIC_SMALLEST;

    // A self-adjoint random dense matrix of size n_dim x n_dim and the
    // corresponding sparse matrix
    MatrixXd dm = MatrixXd::Random(n_dim,n_dim).selfadjointView<Upper>();
    SparseMatrix<double> sm = dm.sparseView();

    // Solve for the eigenvalues
    SymmetricEigenSolver<double> s = Solve(sm, n_ev, type);

    // Output
    std::cout << "Matrix: " << std::endl << std::endl
              << dm << std::endl << std::endl << std::endl
              << "Its " << n_ev << " smallest eigenvalues: "
              << std::endl << std::endl
              << s.eigenvalues() << std::endl << std::endl
              << "And the corresponding eigenvectos: "
              << std::endl << std::endl
              << s.eigenvectors() << std::endl;
}
