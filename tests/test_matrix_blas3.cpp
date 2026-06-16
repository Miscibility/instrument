// test_matrix_blas3.cpp -- unit tests for the DenseMatrix matrix-matrix product
// (CBLAS gemm): multiply, multiply_into, operator*. boost/ut.


#include <boost/ut.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

#include "instrument/matrix.hpp"

namespace mi = miscibility::instrument;

namespace {

template<class T> bool close(T a, T b, T tol = T(1e-12))
{
    return std::abs(a - b) <= tol * (T(1) + std::abs(a) + std::abs(b));
}

// Dense reference: C = A * B from a scalar triple loop (row-major logical view).
std::vector<std::vector<double>> ref_product(const std::vector<std::vector<double>>& a,
                                             const std::vector<std::vector<double>>& b)
{
    const std::size_t m = a.size();
    const std::size_t k = a[0].size();
    const std::size_t n = b[0].size();
    std::vector<std::vector<double>> c(m, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t p = 0; p < k; ++p) {
                c[i][j] += a[i][p] * b[p][j];
            }
        }
    }
    return c;
}

} // namespace

int main()
{
    using namespace boost::ut;

    suite<"DenseMatrixMatMat"> matmat_suite = [] {
        test("multiply: A*B matches the closed form, operator* and a triple loop") = [] {
            mi::DenseMatrix<double> a{{1, 2}, {3, 4}};
            mi::DenseMatrix<double> b{{5, 6}, {7, 8}};
            auto c = a.multiply(b);
            expect(c.rows() == 2_u);
            expect(c.columns() == 2_u);
            expect(close(c(0, 0), 19.0)); // 1*5 + 2*7
            expect(close(c(0, 1), 22.0)); // 1*6 + 2*8
            expect(close(c(1, 0), 43.0)); // 3*5 + 4*7
            expect(close(c(1, 1), 50.0)); // 3*6 + 4*8

            auto c2 = a * b;
            expect(close(c2(0, 0), 19.0));
            expect(close(c2(1, 1), 50.0));
        };

        test("non-square chain: 2x3 times 3x4 gives 2x4, matching a reference loop") = [] {
            std::vector<std::vector<double>> ra{{1, 2, 3}, {4, 5, 6}};
            std::vector<std::vector<double>> rb{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
            mi::DenseMatrix<double> a{{1, 2, 3}, {4, 5, 6}};
            mi::DenseMatrix<double> b{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
            auto c = a.multiply(b);
            const auto rc = ref_product(ra, rb);
            expect(c.rows() == 2_u);
            expect(c.columns() == 4_u);
            for (std::size_t i = 0; i < 2; ++i) {
                for (std::size_t j = 0; j < 4; ++j) {
                    expect(close(c(i, j), rc[i][j])) << "entry (" << i << "," << j << ")";
                }
            }
        };

        test("transposed operand: A^T * B uses A's transpose") = [] {
            // A is 2x2; A^T = {{1,3},{2,4}}. B = {{5,6},{7,8}}.
            mi::DenseMatrix<double> a{{1, 2}, {3, 4}};
            mi::DenseMatrix<double> b{{5, 6}, {7, 8}};
            auto c = a.multiply(b, mi::Transpose::Transposed);
            expect(c.rows() == 2_u);
            expect(c.columns() == 2_u);
            expect(close(c(0, 0), 26.0)); // 1*5 + 3*7
            expect(close(c(0, 1), 30.0)); // 1*6 + 3*8
            expect(close(c(1, 0), 38.0)); // 2*5 + 4*7
            expect(close(c(1, 1), 44.0)); // 2*6 + 4*8
        };

        test("multiply_into with alpha/beta is the scale-add variant") = [] {
            mi::DenseMatrix<double> a{{1, 2}, {3, 4}};
            mi::DenseMatrix<double> b{{5, 6}, {7, 8}};
            mi::DenseMatrix<double> c{{1, 1}, {1, 1}};
            a.multiply_into(b, c, 2.0, 3.0); // 2*A*B + 3*C
            expect(close(c(0, 0), 41.0)); // 41
            expect(close(c(0, 1), 47.0)); // 47
            expect(close(c(1, 0), 89.0)); // 89
            expect(close(c(1, 1), 103.0)); // 103
        };

        test("dimension mismatch throws invalid_argument") = [] {
            mi::DenseMatrix<double> a{{1, 2, 3}, {4, 5, 6}};       // 2x3
            mi::DenseMatrix<double> b{{1, 2, 3, 4}, {5, 6, 7, 8}}; // 2x4 -> inner 3 != 2
            expect(throws<std::invalid_argument>([&] { (void)a.multiply(b); }));

            // Correctly-shaped product but a wrong-sized pre-allocated destination.
            mi::DenseMatrix<double> a2{{1, 2}, {3, 4}};
            mi::DenseMatrix<double> b2{{5, 6}, {7, 8}};
            mi::DenseMatrix<double> wrong_c(3, 3);
            expect(throws<std::invalid_argument>([&] { a2.multiply_into(b2, wrong_c); }));
        };

        test("float and double both work") = [] {
            mi::DenseMatrix<float> af{{1, 2}, {3, 4}};
            mi::DenseMatrix<float> bf{{5, 6}, {7, 8}};
            auto cf = af.multiply(bf);
            expect(close(cf(0, 0), 19.0F));
            expect(close(cf(1, 1), 50.0F));

            mi::DenseMatrix<double> ad{{1, 2}, {3, 4}};
            mi::DenseMatrix<double> bd{{5, 6}, {7, 8}};
            auto cd = ad.multiply(bd);
            expect(close(cd(0, 0), 19.0));
            expect(close(cd(1, 1), 50.0));
        };
    };
}
