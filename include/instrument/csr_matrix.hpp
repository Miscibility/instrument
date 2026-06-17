#pragma once

#include "instrument/matrix.hpp"
#include "instrument/sparsity_pattern.hpp"
#include "instrument/vector.hpp"

#include <taskflow/taskflow.hpp>

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
        (void)pattern;
        throw std::runtime_error{"not implemented"};
    }

    CsrMatrix(const SparsityPattern<T>& pattern, tf::Executor& executor)
        requires(Exec == Execution::Parallel)
    {
        (void)pattern;
        (void)executor;
        throw std::runtime_error{"not implemented"};
    }

    // -- reinitialize ---------------------------------------------------------

    void reinitialize(const SparsityPattern<T>& pattern)
    {
        (void)pattern;
        throw std::runtime_error{"not implemented"};
    }

    // -- compression ----------------------------------------------------------

    void compress(T tolerance = T(0))
    {
        (void)tolerance;
        throw std::runtime_error{"not implemented"};
    }

    // -- dimensions -----------------------------------------------------------

    [[nodiscard]] size_type rows() const noexcept { return rows_; }
    [[nodiscard]] size_type columns() const noexcept { return cols_; }
    [[nodiscard]] size_type nonzeros() const noexcept { return values_.size(); }

    // -- matrix-vector product ------------------------------------------------

    void multiply_into(const Vector<T>& x, Vector<T>& y, T alpha = T(1), T beta = T(0),
                       Transpose op = Transpose::None) const
    {
        (void)x;
        (void)y;
        (void)alpha;
        (void)beta;
        (void)op;
        throw std::runtime_error{"not implemented"};
    }

    [[nodiscard]] Vector<T> multiply(const Vector<T>& x, Transpose op = Transpose::None) const
    {
        (void)x;
        (void)op;
        throw std::runtime_error{"not implemented"};
    }

    [[nodiscard]] Vector<T> operator*(const Vector<T>& x) const
    {
        (void)x;
        throw std::runtime_error{"not implemented"};
    }

private:
    /// @internal Empty placeholder so the serial specialization stores no executor.
    struct no_executor {};

    std::vector<size_type> row_offsets_;   ///< CSR row offsets (length rows_ + 1).
    std::vector<size_type> column_indices_; ///< CSR column indices (length nonzeros()).
    std::vector<T> values_{};                 ///< CSR values (length nonzeros()).
    size_type rows_{0};                       ///< Logical row count.
    size_type cols_{0};                       ///< Logical column count.

    /// @internal Non-owning executor reference, present only in the parallel specialization.
    [[no_unique_address]] std::conditional_t<is_parallel, tf::Executor*, no_executor> executor_{};
};

} // namespace miscibility::instrument
