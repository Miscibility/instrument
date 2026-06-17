/**
 * @file csr_matrix.hpp
 * @brief A sparse matrix in Compressed Sparse Row (CSR) format with a serial or parallel matvec.
 *
 * @code{.cpp}
 * using miscibility::instrument::CsrMatrix;
 * using miscibility::instrument::SparsityPattern;
 * SparsityPattern<double> p(3, 3, {{0, 0, 1}, {0, 2, 5}, {1, 1, 2}, {2, 2, 3}});
 * CsrMatrix<double> a(p);                                 // serial (the default)
 * auto y = a * miscibility::instrument::Vector<double>{1, 1, 1}; // {6, 2, 3}
 * @endcode
 *
 * @par What it is
 * A CsrMatrix owns the three canonical CSR arrays -- `row_offsets` (length `rows()+1`),
 * `column_indices` and `values` (each length `nonzeros()`) -- copied from a @ref SparsityPattern.
 * It is the *only* way to build one: there is no raw-array constructor. The matrix exposes just the
 * matrix-vector seam (@ref MatrixOperator), so it can serve as a block of a block matrix anywhere a
 * @ref DenseMatrix could.
 *
 * @par Serial vs parallel execution
 * The @ref Execution template parameter selects how the forward product walks its rows. The default
 * @ref Execution::Serial runs on the calling thread. @ref Execution::Parallel parallelizes the
 * forward product over rows with Taskflow: a `tf::Executor` is injected at construction and held by
 * *non-owning* reference for the matrix's whole lifetime, so the executor must outlive the matrix
 * (the caller owns and manages it). Because the executor is injected rather than templated into the
 * product, `multiply_into`'s signature is identical across both policies.
 *
 * @par Structural zeros and compress()
 * A CsrMatrix keeps exactly the slots its source @ref SparsityPattern recorded, including explicit
 * structural zeros (a `(row, col)` whose assembled value happened to cancel to zero). Call
 * compress() to drop them; with a positive tolerance it also prunes any genuinely small entries.
 *
 * @par Refilling
 * reinitialize() cheaply rebuilds the matrix from another pattern -- possibly of a different shape;
 * the parallel specialization keeps its existing executor reference across the refill.
 *
 * @par Header-only
 * C++23. Builds on @ref Vector, @ref SparsityPattern and @ref DenseMatrix (for @ref Transpose);
 * the parallel specialization uses Taskflow, which the header always includes.
 */

#pragma once

#include "instrument/matrix.hpp"
#include "instrument/sparsity_pattern.hpp"
#include "instrument/vector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <taskflow/taskflow.hpp>
#include <type_traits>
#include <vector>

namespace miscibility::instrument {

/**
 * @brief Row-execution policy for @ref CsrMatrix's forward matrix-vector product.
 *
 * Selected as a compile-time template argument; the transposed product is always serial.
 */
enum class Execution : std::uint8_t {
    Serial,   ///< Walk the rows on the calling thread (the default).
    Parallel, ///< Parallelize the forward product over rows via an injected `tf::Executor`.
};

/**
 * @brief A sparse matrix in CSR format that models @ref MatrixOperator, built from a @ref SparsityPattern.
 * @tparam T    Element type (an IEEE floating-point @ref Scalar).
 * @tparam Exec Row-execution policy (@ref Execution); defaults to @ref Execution::Serial.
 *
 * Owns the three canonical CSR arrays copied from the pattern. The @ref Execution::Parallel
 * specialization additionally holds a non-owning `tf::Executor*` injected at construction. See the
 * file overview for the execution policies, structural zeros / compress(), and refilling.
 */
template<Scalar T, Execution Exec = Execution::Serial> class CsrMatrix {
public:
    using value_type = T;          ///< Element type.
    using size_type = std::size_t; ///< Dimension / index type.

    /// @brief True iff this is the @ref Execution::Parallel specialization (holds an executor).
    static constexpr bool is_parallel = (Exec == Execution::Parallel);

    // -- construction (the only entry points; build from a SparsityPattern) ---

    /**
     * @brief Build a serial CSR matrix by copying @p pattern's CSR arrays. Serial policy only.
     * @param pattern The assembled sparsity pattern to copy structure and values from.
     */
    explicit CsrMatrix(const SparsityPattern<T>& pattern)
        requires(Exec == Execution::Serial)
    {
        adopt(pattern);
    }

    /**
     * @brief Build a parallel CSR matrix from @p pattern and bind it to @p executor. Parallel only.
     * @param pattern  The assembled sparsity pattern to copy structure and values from.
     * @param executor The Taskflow executor used by every subsequent forward `multiply_into`. Held
     *                 by non-owning reference, so it **must outlive this matrix** (the caller owns it).
     */
    CsrMatrix(const SparsityPattern<T>& pattern, tf::Executor& executor)
        requires(Exec == Execution::Parallel)
        : executor_(&executor)
    {
        adopt(pattern);
    }

    // -- reinitialize ---------------------------------------------------------

    /**
     * @brief Replace this matrix's structure and values with @p pattern's, reusing the object.
     * @param pattern The new sparsity pattern; may have a different shape, which updates rows(),
     *                columns() and nonzeros() accordingly.
     *
     * Cheaper than constructing afresh when refilling a reusable slot. The parallel specialization
     * keeps its existing executor reference.
     */
    void reinitialize(const SparsityPattern<T>& pattern) { adopt(pattern); }

    // -- compression ----------------------------------------------------------

    /**
     * @brief Drop stored slots whose value is at or below @p tolerance in magnitude, in place.
     * @param tolerance Non-negative threshold; a slot is removed when `|value| <= tolerance`. The
     *                  default of `0` removes only exact (structural) zeros, leaving small nonzeros.
     *
     * A @ref SparsityPattern retains explicit structural zeros, so a freshly built CsrMatrix may
     * store entries equal to zero; this prunes them (and, with a positive tolerance, any genuinely
     * small entries) and rebuilds the CSR arrays so nonzeros() reflects the survivors.
     * @code{.cpp}
     * CsrMatrix<double> a(pattern); // pattern has a (0,0) slot that cancelled to 0
     * a.compress();                 // the zero slot is dropped; nonzeros() shrinks
     * a.compress(1e-9);             // also drops any |value| <= 1e-9
     * @endcode
     */
    void compress(T tolerance = T(0))
    {
        std::vector<size_type> new_offsets(rows_ + 1, 0);
        std::vector<size_type> new_columns;
        std::vector<T> new_values;
        new_columns.reserve(column_indices_.size());
        new_values.reserve(values_.size());

        for (size_type i = 0; i < rows_; ++i) {
            for (size_type k = row_offsets_[i]; k < row_offsets_[i + 1]; ++k) {
                if (std::abs(values_[k]) <= tolerance) {
                    continue; // drop slots at or below the tolerance (exact zeros when tolerance == 0)
                }
                new_columns.push_back(column_indices_[k]);
                new_values.push_back(values_[k]);
                ++new_offsets[i + 1];
            }
        }
        for (size_type i = 0; i < rows_; ++i) {
            new_offsets[i + 1] += new_offsets[i];
        }
        row_offsets_ = std::move(new_offsets);
        column_indices_ = std::move(new_columns);
        values_ = std::move(new_values);
    }

    // -- dimensions -----------------------------------------------------------

    [[nodiscard]] size_type rows() const noexcept { return rows_; }    ///< Logical row count.
    [[nodiscard]] size_type columns() const noexcept { return cols_; } ///< Logical column count.
    /// @brief Number of stored slots (length of the values/column arrays). @return The nonzero count.
    [[nodiscard]] size_type nonzeros() const noexcept { return values_.size(); }

    // -- matrix-vector product ------------------------------------------------

    /**
     * @brief Matrix-vector product into a pre-allocated vector: `y <- alpha*op(A)*x + beta*y`.
     * @param x     Right operand. Length `columns()` for @c None, `rows()` for @c Transposed.
     * @param y     Destination, written in place. Length `rows()` for @c None, `columns()` for
     *              @c Transposed; must already be sized.
     * @param alpha Scale applied to the product (default 1).
     * @param beta  Scale applied to the existing @p y before accumulation (default 0).
     * @param op    Whether to use `A` (@c None) or `A^T` (@c Transposed).
     * @throws std::invalid_argument if @p x or @p y has the wrong length for @p op.
     *
     * For @c None the rows are independent (`y_i = alpha * Σ_k values[k]*x[col[k]] + beta*y_i`), so
     * the @ref Execution::Parallel specialization splits them across the executor with no write
     * contention. The @c Transposed path scales @p y by @p beta then scatter-adds into `y[col]`, and
     * is always serial regardless of the execution policy.
     */
    void multiply_into(const Vector<T>& x, Vector<T>& y, T alpha = T(1), T beta = T(0),
                       Transpose op = Transpose::None) const
    {
        const size_type xlen = (op == Transpose::None) ? cols_ : rows_;
        const size_type ylen = (op == Transpose::None) ? rows_ : cols_;
        if (x.size() != xlen || y.size() != ylen) {
            throw std::invalid_argument{"miscibility::instrument::CsrMatrix matrix-vector size mismatch"};
        }

        if (op == Transpose::Transposed) {
            // TODO: decide if I ever want this more efficient
            multiply_transposed(x, y, alpha, beta); // scatter path, serial for both policies
            return;
        }

        // op == None: rows are independent, so each row writes a disjoint y_i.
        if constexpr (is_parallel) {
            multiply_rows_parallel(x, y, alpha, beta);
        }
        else {
            multiply_rows(x, y, alpha, beta, 0, rows_);
        }
    }

    /**
     * @brief Matrix-vector product `op(A)*x`, returning a fresh vector.
     * @param x  Right operand (see multiply_into() for the length rule).
     * @param op Whether to use `A` (@c None) or `A^T` (@c Transposed).
     * @return The product, a new `Vector<T>` of length `rows()` (or `columns()` if @c Transposed).
     * @throws std::invalid_argument if @p x has the wrong length.
     */
    [[nodiscard]] Vector<T> multiply(const Vector<T>& x, Transpose op = Transpose::None) const
    {
        Vector<T> y((op == Transpose::None) ? rows_ : cols_);
        multiply_into(x, y, T(1), T(0), op);
        return y;
    }

    /// @brief `A*x` as a fresh vector. @param x Operand of length `columns()`. @return The product.
    /// @throws std::invalid_argument on a length mismatch.
    [[nodiscard]] Vector<T> operator*(const Vector<T>& x) const { return multiply(x); }

private:
    /// @internal Empty placeholder so the serial specialization stores no executor.
    struct no_executor {};

    /// @internal Copy a pattern's CSR arrays and shape into this matrix (keeps any executor).
    void adopt(const SparsityPattern<T>& pattern)
    {
        rows_ = pattern.rows();
        cols_ = pattern.columns();
        row_offsets_ = pattern.row_offsets();
        column_indices_ = pattern.column_indices();
        values_ = pattern.values();
    }

    /// @internal `y_i <- alpha*(row i . x) + beta*y_i` for rows in `[r0, r1)`.
    void multiply_rows(const Vector<T>& x, Vector<T>& y, T alpha, T beta, size_type r0, size_type r1) const
    {
        for (size_type i = r0; i < r1; ++i) {
            T sum = T(0);
            for (size_type k = row_offsets_[i]; k < row_offsets_[i + 1]; ++k) {
                sum += values_[k] * x[column_indices_[k]];
            }
            y[i] = (alpha * sum) + (beta * y[i]);
        }
    }

    /// @internal Parallel forward product: partition rows into roughly equal-nnz chunks (so skewed
    /// patterns balance), each chunk writing a disjoint slice of `y` -- no synchronization needed.
    void multiply_rows_parallel(const Vector<T>& x, Vector<T>& y, T alpha, T beta) const
    {
        // TODO: See if this is actually faster than serial since there is overhead in
        //       creating the tasks.
        const std::vector<size_type> bounds = chunk_bounds();
        tf::Taskflow flow;
        for (size_type c = 0; c + 1 < bounds.size(); ++c) {
            flow.emplace([this, &x, &y, alpha, beta, r0 = bounds[c], r1 = bounds[c + 1]] {
                multiply_rows(x, y, alpha, beta, r0, r1);
            });
        }
        executor_->run(flow).wait();
    }

    /// @internal Row-index chunk boundaries targeting a small multiple of the worker count, split on
    /// cumulative nnz rather than row count. Always brackets `[0 .. rows_]`.
    [[nodiscard]] std::vector<size_type> chunk_bounds() const
    {
        // TODO: see if I can perform this once per sparsity pattern
        std::vector<size_type> bounds{0};
        const size_type total_nnz = values_.size();
        if (rows_ > 0 && total_nnz > 0) {
            const size_type target = std::max<size_type>(1, executor_->num_workers() * 4);
            const size_type per_chunk = (total_nnz + target - 1) / target; // ceil
            size_type acc = 0;
            for (size_type i = 0; i < rows_; ++i) {
                acc += row_offsets_[i + 1] - row_offsets_[i];
                if (acc >= per_chunk && bounds.back() != i + 1) {
                    bounds.push_back(i + 1);
                    acc = 0;
                }
            }
        }
        if (bounds.back() != rows_) {
            bounds.push_back(rows_);
        }
        return bounds;
    }

    /// @internal `y <- alpha*A^T*x + beta*y`: scale `y` by `beta`, then scatter-add into `y[col]`.
    void multiply_transposed(const Vector<T>& x, Vector<T>& y, T alpha, T beta) const
    {
        for (size_type j = 0; j < cols_; ++j) {
            y[j] = beta * y[j];
        }
        for (size_type i = 0; i < rows_; ++i) {
            const T axi = alpha * x[i];
            for (size_type k = row_offsets_[i]; k < row_offsets_[i + 1]; ++k) {
                y[column_indices_[k]] += values_[k] * axi;
            }
        }
    }

    std::vector<size_type> row_offsets_;    ///< CSR row offsets (length rows_ + 1).
    std::vector<size_type> column_indices_; ///< CSR column indices (length nonzeros()).
    std::vector<T> values_{};               ///< CSR values (length nonzeros()).
    size_type rows_{0};                     ///< Logical row count.
    size_type cols_{0};                     ///< Logical column count.

    /// @internal Non-owning executor reference, present only in the parallel specialization.
    [[no_unique_address]] std::conditional_t<is_parallel, tf::Executor*, no_executor> executor_{};
};

} // namespace miscibility::instrument
