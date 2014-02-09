/*
 * Copyright (c) 2011 Seiya Tokui <beam.web@gmail.com>
 * Copyright (c) 2014 Burkhard Ritter <burkhard@ualberta.ca>
 * This code is distributed under the MIT license.
 *
 * Performance and accuracy of the ARPACK eigensolver compared to the Eigen
 * eigensolver. 
 *
 * Computes execution time of both the ARPACK and the Eigen eigensolver for
 * random symmetric matrices for a range of matrix dimensions. Computes the mean
 * error of the ARPACK eigenvalues and eigenvectors as well. The number of
 * eigenvalues to be computed (for ARPACK) and the number of zeros in each matrix
 * can be specified as a percentage of the matrix dimension. Produces output
 * suitable for plotting.
 *
 * Note: This program uses the Arpaca solver in a very simple, black-box like
 * fashion. It is very likely possible to achieve better performance by tuning
 * various parameters.
 *
 * To compile this program: 
 * g++ \
 *    -std=c++11 \
 *    -I [/path/to/eigen] \
 *    -O3 \
 *    -DNDEBUG \
 *    performance_plot.cpp \
 *    -L [/path/to/libarpack.a] \
 *    -larpack \
 *    -o performance_plot
 *
 * This assumes that arpaca.hpp is in the same directory.
 */

#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "arpaca.hpp"
using namespace Eigen;
using namespace arpaca;
using namespace std::chrono;

std::mt19937 generator(42);

VectorXd diagonalize_random_matrix(int n_dim, double r_ev, double r_zeros)
{
    const int n_ev = r_ev * n_dim;
    int n_zeros = r_zeros * n_dim*n_dim;
    if (n_zeros%2 != 0) n_zeros--;

    /*
     * Create matrix
     *
     * Is there an easier way to get a random symmetric matrix with a specified
     * number of zero elements?
     */
    MatrixXd dm = MatrixXd::Random(n_dim,n_dim).selfadjointView<Upper>();
    std::uniform_int_distribution<int> distribution(0,n_dim-1);
    int n_diag = n_zeros / n_dim;
    if (n_diag%2 != 0) n_diag--;
    int n_rest = n_zeros - n_diag;
    while (n_diag > 0)
    {
        int i = distribution(generator);
        if (dm(i,i) != 0)
        {
            dm(i,i) = 0;
            n_diag--;
        }
    }
    while (n_rest > 0)
    {
        int i = distribution(generator);
        int j = distribution(generator);
        if (i!=j && dm(i,j)!=0)
        {
            dm(i,j) = 0;
            dm(j,i) = 0;
            n_rest -= 2;
        }
    }
    SparseMatrix<double> sm = dm.sparseView();
    assert(sm.nonZeros() == n_dim*n_dim - n_zeros);

    /*
     * Diagonalize Arpaca
     *
     * Arpaca (and ARPACK?) do not seem to support the calculation of all
     * eigenvalues directly. The most primitive way to work around this is to
     * calculate n_ev-1 eigenvalues from the bottom and 1 eigenvalue from the
     * top, and that's what I do here.
     */
    steady_clock::time_point begin_a;
    steady_clock::time_point end_a;
    Eigen::VectorXd eigenvalues_a(n_ev);
    Eigen::MatrixXd eigenvectors_a(n_ev,n_ev);
    SymmetricEigenSolver<double> s_a;
    SymmetricEigenSolver<double> s_a_;
    if (n_ev < n_dim)
    { 
        begin_a = steady_clock::now();
        s_a = Solve(sm, n_ev, ALGEBRAIC_SMALLEST);
        end_a = steady_clock::now();
        // We could avoid copying the vector and matrix by using references.
        eigenvalues_a = s_a.eigenvalues();
        eigenvectors_a = s_a.eigenvectors();
    }
    else // n_ev == n_dim
    {
        begin_a = steady_clock::now();
        s_a = Solve(sm, n_ev-1, ALGEBRAIC_SMALLEST);
        s_a_ = Solve(sm, 1, ALGEBRAIC_LARGEST);
        end_a = steady_clock::now();
        
        eigenvalues_a.head(n_ev-1) = s_a.eigenvalues();
        eigenvectors_a.leftCols(n_ev-1) = s_a.eigenvectors();
        
        eigenvalues_a.tail(1) = s_a_.eigenvalues();
        eigenvectors_a.rightCols(1) = s_a_.eigenvectors();
    }
    duration<double> d_a = duration_cast<duration<double>>(end_a - begin_a);
    double time_a = d_a.count();

    /*
     * Diagonalize Eigen
     */
    Eigen::VectorXd eigenvalues_e(n_ev);
    Eigen::MatrixXd eigenvectors_e(n_ev,n_ev);

    steady_clock::time_point begin_e = steady_clock::now();
    SelfAdjointEigenSolver<MatrixXd> s_e(dm);
    steady_clock::time_point end_e = steady_clock::now();
    
    // We could avoid copying the vector and matrix by using references.
    eigenvalues_e = s_e.eigenvalues().topRows(n_ev);
    eigenvectors_e = s_e.eigenvectors().leftCols(n_ev);
    
    duration<double> d_e = duration_cast<duration<double>>(end_e - begin_e);
    double time_e = d_e.count();

    /*
     * Calculate error
     */
    double error_values = (eigenvalues_a - eigenvalues_e).cwiseAbs().mean();
    double error_vectors = 0;
    for (int i = 0; i < n_ev; i++)
    {
        double d1 = 
            (eigenvectors_a.col(i) - eigenvectors_e.col(i)).cwiseAbs().mean();
        double d2 =
            (eigenvectors_a.col(i) + eigenvectors_e.col(i)).cwiseAbs().mean();
        error_vectors += std::min(d1, d2);
    }
    error_vectors /= n_ev;

    /*
     * Output
     */
    /*
    std::cout << "Matrix: " << std::endl;
    std::cout << dm << std::endl << std::endl;
    std::cout << "Arpack eigenvalues: " << std::endl;
    std::cout << eigenvalues_a << std::endl << std::endl;
    std::cout << "Eigen eigenvalues: " << std::endl;
    std::cout << eigenvalues_e << std::endl << std::endl;
    std::cout << "Time in seconds of Arpaca: " << time_a << std::endl;
    std::cout << "Time in seconds of Eigen: " << time_e << std::endl;
    std::cout << "Mean error of eigenvalues: " << error_values << std::endl;
    std::cout << "Mean error of eigenvectors: " << error_vectors << std::endl;
    std::cout << "info: " << s_a.GetInfo() << std::endl;
    std::cout << "# actual iterations: "
              << s_a.num_actual_iterations() << std::endl;
    std::cout << "# converged eigenvalues: "
              << s_a.num_converged_eigenvalues() << std::endl;
    */

    /*
     * Result
     */
    VectorXd r(6);
    r << time_a, time_e, error_values, error_vectors, 
         s_a.num_actual_iterations(), 
         s_a.num_converged_eigenvalues();
    return r;
}

void print_usage(char* program)
{
    std::cerr 
        << "Performance and accuracy of the ARPACK eigensolver compared "
        << std::endl
        << "to the Eigen eigensolver." << std::endl << std::endl
        << "Computes execution time and error over matrix dimensions. " 
        << std::endl
        << "Produces output suitable for plotting." << std::endl
        << std::endl
        << "Usage: " << std::endl
        << "   " << program
        << " [n_dim_min] [n_dim_max] [n_dp] [n_rep] [r_ev] [r_zeros]"
        << std::endl << std::endl
        << "with: " << std::endl
        << "   n_dim_min: start value for matrix dimension" << std::endl
        << "   n_dim_end: end value for matrix dimension" << std::endl
        << "   n_dp: number of data points to compute" << std::endl
        << "   n_rep: number of repetitions for each matrix dimension" << std::endl
        << "   r_ev: number of eigenvalues to compute as a percentage" << std::endl 
        << "         of matrix dimension, ranges from 0 to 1" << std::endl
        << "   r_zeros: number of zeros in matrix as a percentage of" << std::endl
        << "            matrix size, ranges from 0 to 1" << std::endl
        << std::endl
        << "Example: " << std::endl
        << "   " << program
        << " 100 1000 10 10 0.3 0.1" << std::endl;
}

void print_header(int n_rep, double r_ev, double r_zeros, int w)
{
    std::cout << std::setprecision(6);
    std::cout << std::left;
    std::cout 
        << "# Performance and accuracy of the ARPACK eigensolver compared "
        << "to the Eigen eigensolver." << std::endl
        << "# " << std::endl
        << "# n_rep="  << n_rep << std::endl
        << "# r_ev=" << r_ev << std::endl
        << "# r_zeros=" << r_zeros << std::endl
        << "# " << std::endl
        << "# " 
        << std::setw(w-2) << "n_dim"
        << std::setw(w) << "time_a"
        << std::setw(w) << "time_e"
        << std::setw(w) << "error_values"
        << std::setw(w) << "error_vectors"
        << std::setw(w) << "n_it"
        << std::setw(w) << "n_conv"
        << std::endl;
}

int main (int argc, char** argv)
{
    if (argc != 7)
    {
        print_usage(argv[0]);
        std::exit(EXIT_SUCCESS);
    }
    
    int n_dim_min = std::atoi(argv[1]);
    int n_dim_max = std::atoi(argv[2]);
    int n_dp = std::atoi(argv[3]);
    int n_rep = std::atoi(argv[4]);
    double r_ev = std::atof(argv[5]);
    double r_zeros = std::atof(argv[6]);

    if (n_dim_min < 1) n_dim_min = 1;
    if (n_dim_max < n_dim_min) n_dim_max = n_dim_min;
    if (n_dp < 1) n_dp = 1;
    if (n_rep < 1) n_rep = 1;
    if (r_ev < 0) r_ev = 0;
    if (r_ev > 1) r_ev = 1;
    if (r_zeros < 0) r_zeros = 0;
    if (r_zeros > 1) r_zeros = 1;

    int delta_dim = n_dim_max - n_dim_min + 1;
    if (n_dp > 1)
        delta_dim = (n_dim_max - n_dim_min) / (n_dp-1);
    if (delta_dim == 0) delta_dim = 1;

    const int w = 15;
    print_header(n_rep, r_ev, r_zeros, w);
    for (int n_dim = n_dim_min; n_dim <= n_dim_max; n_dim += delta_dim)
    {
        Eigen::VectorXd v(6);
        v.setZero();
        for (int i=0; i<n_rep; i++)
        {
            VectorXd r = diagonalize_random_matrix(n_dim,r_ev,r_zeros);
            v += r;
        }
        v /= n_rep;
        std::cout << std::setw(w) << n_dim;
        for (int i=0; i<v.size(); i++)
            std::cout << std::setw(w) << v(i);
        std::cout << std::endl;
    }

    return 0;
}
