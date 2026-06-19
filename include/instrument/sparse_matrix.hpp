#pragma once

#include "instrument/array.hpp"
#include "instrument/context.hpp"
#include "instrument/sparsity_pattern.hpp"

#include <algorithm>
#include <cstdint>
#include <ginkgo/ginkgo.hpp>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace miscibility::instrument {

/// Ginkgo sparse storage format a :cpp:`SparseMatrix` can hold.
enum class SparseFormat : std::uint8_t {
    Csr,    ///< Compressed sparse row; the workhorse most solvers expect.
    Coo,    ///< Coordinate list.
    Ell,    ///< ELLPACK (fixed nonzeros per row, padded).
    Sellp,  ///< Sliced ELLPACK.
    Hybrid, ///< ELL + COO split for irregular row lengths.
};

/// Construction options for a :cpp:`SparseMatrix`.
///
/// Only ``format`` is always meaningful; the optional knobs are consulted only
/// for the formats they name and otherwise fall back to Ginkgo's defaults.
struct SparseOptions {
    SparseFormat format = SparseFormat::Csr;
    std::optional<gko::size_type> ell_max_nonzeros_per_row = std::nullopt; ///< Fixed row width for ``Ell``.
    std::optional<gko::size_type> sellp_slice_size = std::nullopt;         ///< Slice size for ``Sellp``.
};

/// A sparse matrix in any of Ginkgo's storage formats, with cheap in-place value refill.
///
/// A ``SparseMatrix`` is assembled from a :cpp:`SparsityPattern` into the format
/// chosen in :cpp:`SparseOptions`, then stored type-erased as the handle's
/// ``LinOp`` — so it drops into solvers, blocks, and raw Ginkgo via the inherited
/// ``apply`` (SpMV).
///
/// .. code-block:: cpp
///
///     SparsityPattern<double> pattern{gko::dim<2>{3, 3}};
///     pattern.add_value(0, 0, 2.0);
///     // ... assemble ...
///     SparseMatrix<double> a{ctx, "A", pattern};   // CSR by default
///     a.linop()->apply(x, y);                       // y = A*x
///
/// The :cpp:`revision` counter and :cpp:`update_values` are the key to reusing a
/// matrix across a Newton / implicit-ODE loop: refilling values in place keeps the
/// same ``LinOp`` pointer but bumps the revision, so a consumer caching work off
/// this matrix can detect that the numbers changed without the pointer changing.
///
/// :tparam T: Floating-point value type.
/// :tparam I: Index type (defaults to ``int``).
template<Scalar T = double, class I = int> class SparseMatrix : public OperatorHandle {
public:
    /// Assembles the pattern into the chosen format.
    ///
    /// :param ctx: Context the matrix lives on and registers with.
    /// :param name: Name reported for this matrix in timing output.
    /// :param pattern: The assembled structure and values to read in.
    /// :param opts: Storage format and format-specific knobs.
    SparseMatrix(Context& ctx, std::string name, const SparsityPattern<T, I>& pattern, SparseOptions opts = {});

    /// The storage format in use.
    [[nodiscard]] SparseFormat format() const noexcept { return format_; }
    /// The number of stored entries.
    [[nodiscard]] gko::size_type num_nonzeros() const;
    /// A counter bumped on every value mutation, so a stale cache can be detected.
    [[nodiscard]] std::uint64_t revision() const noexcept { return revision_; } // TODO: maybe can use a smaller integer type

    /// Refills the values in place from a pattern with the same structure.
    ///
    /// The ``LinOp`` pointer is unchanged; only the numbers and :cpp:`revision` move.
    ///
    /// :param pattern: A pattern with the same nonzero count and layout as this matrix.
    /// :throws std::invalid_argument: if the pattern's nonzero count differs.
    void update_values(const SparsityPattern<T, I>& pattern);

    /// Refills the stored value array directly, in storage order.
    ///
    /// :param values: New values; length must equal :cpp:`num_nonzeros`.
    /// :throws std::invalid_argument: if ``values`` has the wrong length.
    void update_values(std::span<const T> values);

    /// Returns an equivalent matrix in another storage format.
    ///
    /// :param ctx: Context for the new matrix.
    /// :param name: Name for the new matrix.
    /// :param target: The format to convert into.
    [[nodiscard]] SparseMatrix convert_to(Context& ctx, std::string name, SparseFormat target) const;

    /// CSR row pointers (length ``rows + 1``); CSR format only.
    ///
    /// :throws std::runtime_error: if the format is not ``Csr``.
    [[nodiscard]] const I* row_ptrs() const;
    /// CSR column indices (length :cpp:`num_nonzeros`); CSR format only.
    ///
    /// :throws std::runtime_error: if the format is not ``Csr``.
    [[nodiscard]] const I* col_idxs() const;
    /// CSR stored values (length :cpp:`num_nonzeros`); CSR format only.
    ///
    /// :throws std::runtime_error: if the format is not ``Csr``.
    [[nodiscard]] const T* values() const;
    /// Mutable CSR stored values, for the refill-in-place fast path; CSR format only.
    ///
    /// Writing through this pointer does not bump :cpp:`revision`; prefer
    /// :cpp:`update_values` when a consumer relies on the revision counter.
    ///
    /// :throws std::runtime_error: if the format is not ``Csr``.
    [[nodiscard]] T* values();

private:
    using csr_type = gko::matrix::Csr<T, I>;
    using coo_type = gko::matrix::Coo<T, I>;
    using ell_type = gko::matrix::Ell<T, I>;
    using sellp_type = gko::matrix::Sellp<T, I>;
    using hybrid_type = gko::matrix::Hybrid<T, I>;

    SparseMatrix(Context& ctx, std::string name, std::shared_ptr<gko::LinOp> matrix, SparseFormat format);

    static std::shared_ptr<gko::LinOp> read_format(const std::shared_ptr<const gko::Executor>& exec,
                                                   SparseFormat format, const gko::matrix_data<T, I>& data);

    [[nodiscard]] csr_type* require_csr() const;

    // Calls fn with the stored matrix recovered as its concrete format type.
    template<class F> decltype(auto) visit(F&& fn) const
    {
        gko::LinOp* op = linop().get();
        switch (format_) {
        case SparseFormat::Csr:
            return fn(gko::as<csr_type>(op));
        case SparseFormat::Coo:
            return fn(gko::as<coo_type>(op));
        case SparseFormat::Ell:
            return fn(gko::as<ell_type>(op));
        case SparseFormat::Sellp:
            return fn(gko::as<sellp_type>(op));
        case SparseFormat::Hybrid:
            return fn(gko::as<hybrid_type>(op));
        }
        throw std::logic_error{"miscibility::instrument::SparseMatrix unknown format"};
    }

    SparseFormat format_;
    std::uint64_t revision_ = 0;
};

template<Scalar T, class I>
SparseMatrix<T, I>::SparseMatrix(Context& ctx, std::string name, std::shared_ptr<gko::LinOp> matrix,
                                 SparseFormat format) :
    OperatorHandle(ctx, std::move(name), std::move(matrix)), format_{format}
{
}

template<Scalar T, class I>
std::shared_ptr<gko::LinOp> SparseMatrix<T, I>::read_format(const std::shared_ptr<const gko::Executor>& exec,
                                                            SparseFormat format, const gko::matrix_data<T, I>& data)
{
    switch (format) {
    case SparseFormat::Csr: {
        auto matrix = csr_type::create(exec);
        matrix->read(data);
        return gko::share(std::move(matrix));
    }
    case SparseFormat::Coo: {
        auto matrix = coo_type::create(exec);
        matrix->read(data);
        return gko::share(std::move(matrix));
    }
    case SparseFormat::Ell: {
        auto matrix = ell_type::create(exec);
        matrix->read(data);
        return gko::share(std::move(matrix));
    }
    case SparseFormat::Sellp: {
        auto matrix = sellp_type::create(exec);
        matrix->read(data);
        return gko::share(std::move(matrix));
    }
    case SparseFormat::Hybrid: {
        auto matrix = hybrid_type::create(exec);
        matrix->read(data);
        return gko::share(std::move(matrix));
    }
    }
    throw std::logic_error{"miscibility::instrument::SparseMatrix unknown format"};
}

template<Scalar T, class I>
SparseMatrix<T, I>::SparseMatrix(Context& ctx, std::string name, const SparsityPattern<T, I>& pattern,
                                 SparseOptions opts) :
    SparseMatrix(ctx, std::move(name), read_format(ctx.executor(), opts.format, pattern.to_matrix_data()), opts.format)
{
}

template<Scalar T, class I> gko::size_type SparseMatrix<T, I>::num_nonzeros() const
{
    return visit([](const auto* matrix) { return matrix->get_num_stored_elements(); });
}

template<Scalar T, class I> void SparseMatrix<T, I>::update_values(const SparsityPattern<T, I>& pattern)
{
    const gko::matrix_data<T, I> data = pattern.to_matrix_data();
    if (data.nonzeros.size() != num_nonzeros()) {
        throw std::invalid_argument{"miscibility::instrument::SparseMatrix::update_values structure mismatch"};
    }
    visit([&](auto* matrix) { matrix->read(data); });
    ++revision_;
}

template<Scalar T, class I> void SparseMatrix<T, I>::update_values(std::span<const T> values)
{
    if (values.size() != num_nonzeros()) {
        throw std::invalid_argument{"miscibility::instrument::SparseMatrix::update_values length mismatch"};
    }
    visit([&](auto* matrix) {
        if constexpr (requires { matrix->get_values(); }) {
            std::copy_n(values.data(), values.size(), matrix->get_values());
        }
        else {
            throw std::runtime_error{"miscibility::instrument::SparseMatrix value refill unsupported for this format"};
        }
    });
    ++revision_;
}

template<Scalar T, class I>
SparseMatrix<T, I> SparseMatrix<T, I>::convert_to(Context& ctx, std::string name, SparseFormat target) const
{
    gko::matrix_data<T, I> data;
    visit([&](const auto* matrix) { matrix->write(data); });
    return SparseMatrix{ctx, std::move(name), read_format(ctx.executor(), target, data), target};
}

template<Scalar T, class I> typename SparseMatrix<T, I>::csr_type* SparseMatrix<T, I>::require_csr() const
{
    if (format_ != SparseFormat::Csr) {
        throw std::runtime_error{"miscibility::instrument::SparseMatrix array accessor requires CSR format"};
    }
    return gko::as<csr_type>(linop().get());
}

template<Scalar T, class I> const I* SparseMatrix<T, I>::row_ptrs() const
{
    return require_csr()->get_const_row_ptrs();
}
template<Scalar T, class I> const I* SparseMatrix<T, I>::col_idxs() const
{
    return require_csr()->get_const_col_idxs();
}
template<Scalar T, class I> const T* SparseMatrix<T, I>::values() const { return require_csr()->get_const_values(); }
template<Scalar T, class I> T* SparseMatrix<T, I>::values() { return require_csr()->get_values(); }

} // namespace miscibility::instrument
