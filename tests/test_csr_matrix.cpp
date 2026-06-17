// test_csr_matrix.cpp -- unit tests for CsrMatrix: construction from a
// SparsityPattern, the gemv matrix-vector seam (None and Transposed), the
// rectangular and empty-row cases, reinitialize, compress, and the
// serial/parallel equivalence of the Execution policy. boost/ut.
//
// Every member currently throws std::runtime_error{"not implemented"}, so each
// runtime test below fails until tdd-3-implement fills them in. The compile-time
// static_asserts pin the MatrixOperator shape for both Execution policies and
// must hold for the suite to compile.

#include "instrument/csr_matrix.hpp"
#include "instrument/matrix.hpp"
#include "instrument/sparsity_pattern.hpp"
#include "instrument/vector.hpp"

#include <boost/ut.hpp>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace mi = miscibility::instrument;

// A CsrMatrix models the matrix-vector seam, for both Execution policies.
static_assert(mi::MatrixOperator<mi::CsrMatrix<double>, double>);
static_assert(mi::MatrixOperator<mi::CsrMatrix<double, mi::Execution::Parallel>, double>);
static_assert(mi::MatrixOperator<mi::CsrMatrix<float>, float>);
static_assert(mi::MatrixOperator<mi::CsrMatrix<float, mi::Execution::Parallel>, float>);

namespace {

template<class T> T default_tol() { return std::is_same_v<T, float> ? T(1e-4) : T(1e-10); }

template<class T> bool close(T a, T b, T tol = default_tol<T>())
{
    return std::abs(a - b) <= tol * (T(1) + std::abs(a) + std::abs(b));
}

template<class T> bool close_vec(const mi::Vector<T>& a, const mi::Vector<T>& b, T tol = default_tol<T>())
{
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!close(a[i], b[i], tol)) {
            return false;
        }
    }
    return true;
}

// Build the equivalent dense matrix from a pattern, for cross-checking.
template<class T> mi::DenseMatrix<T> to_dense(const mi::SparsityPattern<T>& p)
{
    mi::DenseMatrix<T> d(p.rows(), p.columns());
    const auto& ro = p.row_offsets();
    const auto& ci = p.column_indices();
    const auto& vs = p.values();
    for (std::size_t i = 0; i < p.rows(); ++i) {
        for (std::size_t k = ro[i]; k < ro[i + 1]; ++k) {
            d(i, ci[k]) += vs[k];
        }
    }
    return d;
}

// Construct a CsrMatrix of the requested Execution policy; the executor is only
// used (and must outlive the matrix) for the parallel specialization.
template<class T, mi::Execution Exec>
mi::CsrMatrix<T, Exec> make_csr(const mi::SparsityPattern<T>& p, tf::Executor& exec)
{
    if constexpr (Exec == mi::Execution::Serial) {
        (void)exec;
        return mi::CsrMatrix<T, Exec>(p);
    }
    else {
        return mi::CsrMatrix<T, Exec>(p, exec);
    }
}

// The value-level scenarios, parameterized over element type and Execution so the
// serial and parallel paths are exercised in lockstep.
template<class T, mi::Execution Exec> void register_value_tests(const std::string& tag)
{
    using namespace boost::ut;

    test("happy path: 3x3 (diag + one off-diagonal) matches a hand computation " + tag) = [] {
        // A = [[1,0,5],[0,2,0],[0,0,3]]
        mi::SparsityPattern<T> p(3, 3,
                                 {{.row = 0, .col = 0, .value = T(1)},
                                  {.row = 0, .col = 2, .value = T(5)},
                                  {.row = 1, .col = 1, .value = T(2)},
                                  {.row = 2, .col = 2, .value = T(3)}});
        tf::Executor exec;
        auto a = make_csr<T, Exec>(p, exec);
        expect(a.rows() == 3_u);
        expect(a.columns() == 3_u);
        expect(a.nonzeros() == 4_u);
        mi::Vector<T> x{T(1), T(1), T(1)};
        auto y = a.multiply(x);
        expect(y.size() == 3_u);
        expect(close(y[0], T(6))); // 1*1 + 5*1
        expect(close(y[1], T(2))); // 2*1
        expect(close(y[2], T(3))); // 3*1
        auto y2 = a * x;
        expect(close_vec(y2, y));
    };

    test("matches DenseMatrix for a sparse pattern with an empty row " + tag) = [] {
        // 4x4, row 1 entirely empty.
        mi::SparsityPattern<T> p(4, 4,
                                 {{.row = 0, .col = 0, .value = T(2)},
                                  {.row = 0, .col = 3, .value = T(1)},
                                  {.row = 2, .col = 1, .value = T(4)},
                                  {.row = 3, .col = 3, .value = T(5)}});
        tf::Executor exec;
        auto a = make_csr<T, Exec>(p, exec);
        auto dense = to_dense(p);
        mi::Vector<T> x{T(1), T(2), T(3), T(4)};
        expect(close_vec(a.multiply(x), dense.multiply(x)));
    };

    test("gemv: multiply_into(x, y, alpha=2, beta=3) computes 2*A*x + 3*y " + tag) = [] {
        mi::SparsityPattern<T> p(3, 3,
                                 {{.row = 0, .col = 0, .value = T(1)},
                                  {.row = 0, .col = 2, .value = T(5)},
                                  {.row = 1, .col = 1, .value = T(2)},
                                  {.row = 2, .col = 2, .value = T(3)}});
        tf::Executor exec;
        auto a = make_csr<T, Exec>(p, exec);
        mi::Vector<T> x{T(1), T(1), T(1)};
        mi::Vector<T> y{T(1), T(1), T(1)};
        a.multiply_into(x, y, T(2), T(3)); // 2*{6,2,3} + 3*{1,1,1}
        expect(close(y[0], T(15)));
        expect(close(y[1], T(7)));
        expect(close(y[2], T(9)));
    };

    test("transposed matches DenseMatrix, including beta != 0 " + tag) = [] {
        mi::SparsityPattern<T> p(4, 4,
                                 {{.row = 0, .col = 0, .value = T(2)},
                                  {.row = 0, .col = 3, .value = T(1)},
                                  {.row = 2, .col = 1, .value = T(4)},
                                  {.row = 3, .col = 3, .value = T(5)}});
        tf::Executor exec;
        auto a = make_csr<T, Exec>(p, exec);
        auto dense = to_dense(p);
        // Transposed: x has length rows()==4, y has length columns()==4.
        mi::Vector<T> x{T(1), T(2), T(3), T(4)};
        expect(close_vec(a.multiply(x, mi::Transpose::Transposed), dense.multiply(x, mi::Transpose::Transposed)));

        mi::Vector<T> y_csr{T(7), T(8), T(9), T(10)};
        mi::Vector<T> y_dense{T(7), T(8), T(9), T(10)};
        a.multiply_into(x, y_csr, T(2), T(3), mi::Transpose::Transposed);
        dense.multiply_into(x, y_dense, T(2), T(3), mi::Transpose::Transposed);
        expect(close_vec(y_csr, y_dense));
    };

    test("rectangular 2x4: forward length-4 -> length-2, transposed length-2 -> length-4 " + tag) = [] {
        mi::SparsityPattern<T> p(2, 4,
                                 {{.row = 0, .col = 0, .value = T(1)},
                                  {.row = 0, .col = 3, .value = T(2)},
                                  {.row = 1, .col = 1, .value = T(3)}});
        tf::Executor exec;
        auto a = make_csr<T, Exec>(p, exec);
        auto dense = to_dense(p);
        expect(a.rows() == 2_u);
        expect(a.columns() == 4_u);

        mi::Vector<T> x{T(1), T(2), T(3), T(4)};
        auto y = a.multiply(x);
        expect(y.size() == 2_u);
        expect(close_vec(y, dense.multiply(x)));

        mi::Vector<T> xt{T(5), T(6)};
        auto yt = a.multiply(xt, mi::Transpose::Transposed);
        expect(yt.size() == 4_u);
        expect(close_vec(yt, dense.multiply(xt, mi::Transpose::Transposed)));
    };

    test("empty rows produce exactly beta*y_i (zero when beta==0) " + tag) = [] {
        // Row 1 is empty.
        mi::SparsityPattern<T> p(3, 3, {{.row = 0, .col = 0, .value = T(2)}, {.row = 2, .col = 2, .value = T(3)}});
        tf::Executor exec;
        auto a = make_csr<T, Exec>(p, exec);
        mi::Vector<T> x{T(1), T(1), T(1)};

        mi::Vector<T> y{T(10), T(20), T(30)};
        a.multiply_into(x, y, T(1), T(3)); // empty row 1 -> 3*20
        expect(close(y[1], T(60)));

        auto y0 = a.multiply(x); // beta == 0 -> empty row is exactly 0
        expect(close(y0[1], T(0)));
    };

    test("all-zeros pattern (nonzeros()==0): product is beta*y " + tag) = [] {
        mi::SparsityPattern<T> p(3, 3, {});
        tf::Executor exec;
        auto a = make_csr<T, Exec>(p, exec);
        expect(a.nonzeros() == 0_u);
        mi::Vector<T> x{T(1), T(2), T(3)};
        auto y = a.multiply(x);
        expect(close(y[0], T(0)));
        expect(close(y[1], T(0)));
        expect(close(y[2], T(0)));

        mi::Vector<T> y2{T(4), T(5), T(6)};
        a.multiply_into(x, y2, T(1), T(2)); // beta*y
        expect(close(y2[0], T(8)));
        expect(close(y2[1], T(10)));
        expect(close(y2[2], T(12)));
    };

    test("reinitialize with a same-shape pattern updates the values " + tag) = [] {
        mi::SparsityPattern<T> p1(2, 2, {{.row = 0, .col = 0, .value = T(1)}, {.row = 1, .col = 1, .value = T(2)}});
        tf::Executor exec;
        auto a = make_csr<T, Exec>(p1, exec);
        mi::Vector<T> x{T(1), T(1)};
        expect(close(a.multiply(x)[0], T(1)));

        mi::SparsityPattern<T> p2(2, 2, {{.row = 0, .col = 0, .value = T(7)}, {.row = 1, .col = 1, .value = T(9)}});
        a.reinitialize(p2);
        expect(a.rows() == 2_u);
        expect(a.columns() == 2_u);
        auto y = a.multiply(x);
        expect(close(y[0], T(7)));
        expect(close(y[1], T(9)));
    };

    test("reinitialize with a different-shape pattern changes shape and structure " + tag) = [] {
        mi::SparsityPattern<T> p1(2, 2, {{.row = 0, .col = 0, .value = T(1)}, {.row = 1, .col = 1, .value = T(2)}});
        tf::Executor exec;
        auto a = make_csr<T, Exec>(p1, exec);

        mi::SparsityPattern<T> p2(2, 4,
                                  {{.row = 0, .col = 0, .value = T(1)},
                                   {.row = 0, .col = 3, .value = T(2)},
                                   {.row = 1, .col = 1, .value = T(3)}});
        a.reinitialize(p2);
        expect(a.rows() == 2_u);
        expect(a.columns() == 4_u);
        expect(a.nonzeros() == 3_u);
        auto dense = to_dense(p2);
        mi::Vector<T> x{T(1), T(2), T(3), T(4)};
        expect(close_vec(a.multiply(x), dense.multiply(x)));
    };

    test("wrong-length x or y throws invalid_argument (None) " + tag) = [] {
        mi::SparsityPattern<T> p(2, 3, {{.row = 0, .col = 0, .value = T(1)}, {.row = 1, .col = 2, .value = T(2)}});
        tf::Executor exec;
        auto a = make_csr<T, Exec>(p, exec);
        mi::Vector<T> bad_x{T(1), T(1)}; // need length 3
        expect(throws<std::invalid_argument>([&] { (void)a.multiply(bad_x); }));
        mi::Vector<T> x{T(1), T(1), T(1)};
        mi::Vector<T> bad_y{T(0), T(0), T(0)}; // need length 2
        expect(throws<std::invalid_argument>([&] { a.multiply_into(x, bad_y); }));
    };

    test("wrong-length x or y throws invalid_argument (Transposed) " + tag) = [] {
        mi::SparsityPattern<T> p(2, 3, {{.row = 0, .col = 0, .value = T(1)}, {.row = 1, .col = 2, .value = T(2)}});
        tf::Executor exec;
        auto a = make_csr<T, Exec>(p, exec);
        // Transposed: x must have length rows()==2, y length columns()==3.
        mi::Vector<T> bad_x{T(1), T(1), T(1)};
        expect(throws<std::invalid_argument>([&] { (void)a.multiply(bad_x, mi::Transpose::Transposed); }));
        mi::Vector<T> x{T(1), T(1)};
        mi::Vector<T> bad_y{T(0), T(0)};
        expect(
            throws<std::invalid_argument>([&] { a.multiply_into(x, bad_y, T(1), T(0), mi::Transpose::Transposed); }));
    };
}

// Build a banded (tridiagonal) pattern of order n for the parallel-equivalence test.
mi::SparsityPattern<double> banded(std::size_t n)
{
    std::vector<mi::MatrixEntry<double>> entries;
    for (std::size_t i = 0; i < n; ++i) {
        entries.push_back({.row = i, .col = i, .value = 2.0});
        if (i > 0) {
            entries.push_back({.row = i, .col = i - 1, .value = -1.0});
        }
        if (i + 1 < n) {
            entries.push_back({.row = i, .col = i + 1, .value = -1.0});
        }
    }
    return {n, n, std::span<const mi::MatrixEntry<double>>{entries.data(), entries.size()}};
}

} // namespace

int main()
{
    using namespace boost::ut;

    suite<"CsrMatrix"> csr_suite = [] {
        register_value_tests<double, mi::Execution::Serial>("[double, serial]");
        register_value_tests<double, mi::Execution::Parallel>("[double, parallel]");
        register_value_tests<float, mi::Execution::Serial>("[float, serial]");
        register_value_tests<float, mi::Execution::Parallel>("[float, parallel]");

        // -- serial / parallel equivalence on a larger matrix ----------------

        test("parallel matches serial on a 1000x1000 banded matrix (None and Transposed)") = [] {
            constexpr std::size_t n = 1000;
            auto p = banded(n);
            mi::CsrMatrix<double, mi::Execution::Serial> serial(p);

            tf::Executor exec;
            mi::CsrMatrix<double, mi::Execution::Parallel> parallel(p, exec);

            mi::Vector<double> x(n);
            for (std::size_t i = 0; i < n; ++i) {
                x[i] = (static_cast<double>(i) * 0.5) - 3.0;
            }
            expect(close_vec(serial.multiply(x), parallel.multiply(x)));
            expect(close_vec(serial.multiply(x, mi::Transpose::Transposed),
                             parallel.multiply(x, mi::Transpose::Transposed)));
        };

        // -- compress --------------------------------------------------------

        test("compress() with default tolerance removes structural zeros only") = [] {
            // (0,0) coalesces to a structural zero; (1,1) is a genuine nonzero.
            mi::SparsityPattern<double> p(2, 2,
                                          {{.row = 0, .col = 0, .value = 1.0},
                                           {.row = 0, .col = 0, .value = -1.0},
                                           {.row = 1, .col = 1, .value = 5.0}});
            mi::CsrMatrix<double> a(p);
            expect(a.nonzeros() == 2_u); // structural zero retained from the pattern
            mi::Vector<double> x{1.0, 1.0};
            auto before = a.multiply(x);

            a.compress(); // tolerance 0: drop entries with |value| <= 0, i.e. exact zeros
            expect(a.nonzeros() == 1_u);
            // The product is unchanged: the dropped slot held a zero.
            expect(close_vec(a.multiply(x), before));
            expect(close(a.multiply(x)[1], 5.0));
        };

        test("compress(tol) drops stored values with |value| <= tol and changes the product") = [] {
            mi::SparsityPattern<double> p(2, 2,
                                          {{.row = 0, .col = 0, .value = 1e-3}, {.row = 1, .col = 1, .value = 5.0}});
            mi::CsrMatrix<double> a(p);
            expect(a.nonzeros() == 2_u);
            mi::Vector<double> x{1.0, 1.0};
            expect(close(a.multiply(x)[0], 1e-3));

            a.compress(1e-2); // 1e-3 <= 1e-2 -> dropped
            expect(a.nonzeros() == 1_u);
            auto y = a.multiply(x);
            expect(close(y[0], 0.0)); // the small entry is gone
            expect(close(y[1], 5.0));
        };

        test("compress() with default tolerance keeps a small but nonzero value") = [] {
            mi::SparsityPattern<double> p(2, 2,
                                          {{.row = 0, .col = 0, .value = 1e-3}, {.row = 1, .col = 1, .value = 5.0}});
            mi::CsrMatrix<double> a(p);
            a.compress(); // tolerance 0 only drops exact zeros
            expect(a.nonzeros() == 2_u);
            mi::Vector<double> x{1.0, 1.0};
            expect(close(a.multiply(x)[0], 1e-3));
        };

        test("compress() on the parallel specialization behaves identically") = [] {
            mi::SparsityPattern<double> p(2, 2,
                                          {{.row = 0, .col = 0, .value = 1.0},
                                           {.row = 0, .col = 0, .value = -1.0},
                                           {.row = 1, .col = 1, .value = 5.0}});
            tf::Executor exec;
            mi::CsrMatrix<double, mi::Execution::Parallel> a(p, exec);
            expect(a.nonzeros() == 2_u);
            a.compress();
            expect(a.nonzeros() == 1_u);
            mi::Vector<double> x{1.0, 1.0};
            expect(close(a.multiply(x)[1], 5.0));
        };
    };
}
