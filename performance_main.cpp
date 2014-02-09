/*
 * Copyright (c) 2011 Seiya Tokui <beam.web@gmail.com>
 * Copyright (c) 2014 Burkhard Ritter <burkhard@ualberta.ca>
 * This code is distributed under the MIT license.
 *
 * Performance test for Arpaca's eigensolver.
 *
 * To compile this program: 
 * g++ \
 *    -std=c++11 \
 *    -I [/path/to/eigen] \
 *    -O3 \
 *    -DNDEBUG \
 *    performance_main.cpp \
 *    -L [/path/to/libarpack.a] \
 *    -larpack \
 *    -o performance_main
 *
 * This assumes that arpaca.hpp is in the same directory.
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "arpaca.hpp"

namespace chrono = std::chrono;

typedef Eigen::Triplet<double> T;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SMatrix;

template<typename RandomNumberGenerator>
SMatrix
MakeSparseSymmetricRandomMatrix(int n, int k, RandomNumberGenerator& rnd)
{
    std::uniform_int_distribution<int> r_int(0,n-1);
    std::normal_distribution<double> r_real;

    std::vector<T> ts;
    ts.reserve(n*k);
    for (int l=0; l<n*k; l++)
    {
        // If we randomly create two or more triplets for the same matrix
        // element, the values of all triplets will be summed...
        int i = r_int(rnd);
        int j = r_int(rnd);
        double v = r_real(rnd);
        ts.push_back(T(i,j,v));
    }

    SMatrix mat(n, n);
    mat.setFromTriplets(ts.begin(),ts.end());
    return mat.selfadjointView<Eigen::Upper>();
}

arpaca::EigenvalueType
GetEigenvalueType(const std::string& name)
{
    if (name == "LA")
        return arpaca::ALGEBRAIC_LARGEST;
    else if (name == "SA")
        return arpaca::ALGEBRAIC_SMALLEST;
    else if (name == "BE")
        return arpaca::ALGEBRAIC_BOTH_END;
    else if (name == "LM")
        return arpaca::MAGNITUDE_LARGEST;
    else if (name == "SM")
        return arpaca::MAGNITUDE_SMALLEST;
    throw std::invalid_argument("invalid eigenvalue type");
}

int main(int argc, char** argv)
{
    if (argc != 5) {
        std::cerr << "usage: " << argv[0]
                  << " <dimension>"
                  << " <# of non-zero values in each row>"
                  << " <# of eigenvectors>"
                  << " <type of eigenvalues>"
                  << std::endl;
        std::cerr << "\ttype of eigenvalues: LA SA BE LM SM" << std::endl;
        return 1;
    }

    const int n = std::atoi(argv[1]),
    k = std::atoi(argv[2]),
    r = std::atoi(argv[3]);
    const arpaca::EigenvalueType type = GetEigenvalueType(argv[4]);

    std::cerr << "Making matrix" << std::endl;
    std::mt19937 generator(42);
    SMatrix X = MakeSparseSymmetricRandomMatrix(n, k, generator);

    std::cerr << "Start performance test" << std::endl;
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    arpaca::SymmetricEigenSolver<double> solver = arpaca::Solve(X, r, type);

    chrono::steady_clock::time_point end = chrono::steady_clock::now();

    chrono::duration<double> duration_ = 
        chrono::duration_cast<chrono::duration<double>>(end - begin);
    const double duration = duration_.count();

    std::cout << "        DIMENSION: " << X.rows() << std::endl;
    std::cout << "         NONZEROS: " << X.nonZeros() << std::endl;
    std::cout << "         DURATION: "
              << duration << " SEC." << std::endl;
    std::cout << "             ITER: "
              << solver.num_actual_iterations() << std::endl;
    std::cout << "CONVERGED EIGVALS: "
              << solver.num_converged_eigenvalues() << std::endl;
    std::cout << "             INFO: " << solver.GetInfo() << std::endl;
}
