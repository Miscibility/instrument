#pragma once

#include "instrument/matrix.hpp"
#include "instrument/sparsity_pattern.hpp"
#include "instrument/vector.hpp"

#include <taskflow/taskflow.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace miscibility::instrument {

enum class Execution : std::uint8_t {
    Serial,
    Parallel,
};

template<Scalar T, Execution Exec = Execution::Serial> class CsrMatrix {
public:
    using value_type = T;
    using size_type = std::size_t;

    static constexpr bool is_parallel = (Exec == Execution::Parallel);

    // -- construction (the only entry points; build from a SparsityPattern) ---

    explicit CsrMatrix(const SparsityPattern<T>& pattern)
        requires(Exec == Execution::Serial)
    {
        adopt(pattern);
    }

    CsrMatrix(const SparsityPattern<T>& pattern, tf::Executor& executor)
        requires(Exec == Execution::Parallel)
        : executor_(&executor)
    {
        adopt(pattern);
    }

    // -- reinitialize ---------------------------------------------------------

    void reinitialize(const SparsityPattern<T>& pattern) { adopt(pattern); }

    // -- compression ----------------------------------------------------------

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

    [[nodiscard]] size_type rows() const noexcept { return rows_; }
    [[nodiscard]] size_type columns() const noexcept { return cols_; }
    [[nodiscard]] size_type nonzeros() const noexcept { return values_.size(); }

    // -- matrix-vector product ------------------------------------------------

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

    [[nodiscard]] Vector<T> multiply(const Vector<T>& x, Transpose op = Transpose::None) const
    {
        Vector<T> y((op == Transpose::None) ? rows_ : cols_);
        multiply_into(x, y, T(1), T(0), op);
        return y;
    }

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

    std::vector<size_type> row_offsets_;   ///< CSR row offsets (length rows_ + 1).
    std::vector<size_type> column_indices_; ///< CSR column indices (length nonzeros()).
    std::vector<T> values_{};                 ///< CSR values (length nonzeros()).
    size_type rows_{0};                       ///< Logical row count.
    size_type cols_{0};                       ///< Logical column count.

    /// @internal Non-owning executor reference, present only in the parallel specialization.
    [[no_unique_address]] std::conditional_t<is_parallel, tf::Executor*, no_executor> executor_{};
};

} // namespace miscibility::instrument
