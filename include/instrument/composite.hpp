#pragma once

#include "instrument/array.hpp"
#include "instrument/context.hpp"

#include <ginkgo/ginkgo.hpp>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace miscibility::instrument {

namespace detail {

struct AdoptedHandle : OperatorHandle {
    AdoptedHandle(Context& ctx, std::string name, std::shared_ptr<gko::LinOp> op) :
        OperatorHandle(ctx, std::move(name), std::move(op))
    {
    }
};

} // namespace detail

/// Wraps an arbitrary ``LinOp`` as an :cpp:`OperatorHandle`, registering its name with ``ctx``.
///
/// This is the seam the composite builders use to surface a freshly built Ginkgo
/// operator as a handle; it is also handy for adopting any other raw ``LinOp``.
///
/// :param ctx: Context the handle binds to and registers with.
/// :param name: Name to register and time the operator under.
/// :param op: The operator to wrap.
inline OperatorHandle make_operator_handle(Context& ctx, std::string name, std::shared_ptr<gko::LinOp> op)
{
    return detail::AdoptedHandle{ctx, std::move(name), std::move(op)};
}

// TODO: Make sure a linear solver can also be put into this hierarchy

/// A block matrix assembled from a grid of sub-operators.
///
/// The blocks are given row-major; a ``nullptr`` block is a structural zero. The
/// sub-operators may be heterogeneous concrete types — because any wrapper
/// converts implicitly to ``shared_ptr<const gko::LinOp>``, a ``SparseMatrix``, a
/// ``DenseMatrix``, and a ``gko::matrix::Diagonal`` can sit side by side.
///
/// .. code-block:: cpp
///
///     BlockMatrix block{ctx, "M", {{a, b}, {c, d}}};   // 2x2 grid
///     block.linop()->apply(x, y);                       // applies block-wise
///
/// The grid must be consistent: every block-row shares a height, every
/// block-column a width, and no block-row or block-column is entirely null.
class BlockMatrix : public OperatorHandle {
public:
    using size_type = gko::size_type;

    // TODO: Make a convenience type: using ZeroMatrix = nullptr

    /// Builds a block operator from a row-major grid of sub-operators.
    ///
    /// :param ctx: Context the block matrix lives on and registers with.
    /// :param name: Name reported for this operator in timing output.
    /// :param blocks: Row-major grid of blocks; ``nullptr`` marks a zero block.
    /// :throws gko::Error: if the block heights/widths are inconsistent or a
    ///     block-row or block-column is entirely null.
    BlockMatrix(Context& ctx, std::string name,
                std::initializer_list<std::initializer_list<std::shared_ptr<const gko::LinOp>>> blocks);

    /// The grid shape as ``{block_rows, block_cols}`` (not the element dimensions).
    [[nodiscard]] gko::dim<2> block_size() const;

    /// The sub-operator at grid cell ``(i, j)``, or ``nullptr`` for a zero block.
    [[nodiscard]] const gko::LinOp* block_at(size_type i, size_type j) const;
};

namespace detail {

inline std::shared_ptr<gko::LinOp>
make_block_operator(const std::shared_ptr<const gko::Executor>& exec,
                    std::initializer_list<std::initializer_list<std::shared_ptr<const gko::LinOp>>> blocks)
{
    std::vector<std::vector<std::shared_ptr<const gko::LinOp>>> grid;
    grid.reserve(blocks.size());
    for (const auto& row : blocks) {
        grid.emplace_back(row);
    }
    return gko::share(gko::BlockOperator::create(exec, std::move(grid)));
}

} // namespace detail

inline BlockMatrix::BlockMatrix(
    Context& ctx, std::string name,
    std::initializer_list<std::initializer_list<std::shared_ptr<const gko::LinOp>>> blocks) :
    OperatorHandle(ctx, std::move(name), detail::make_block_operator(ctx.executor(), blocks))
{
}

inline gko::dim<2> BlockMatrix::block_size() const
{
    return gko::as<gko::BlockOperator>(linop().get())->get_block_size();
}

inline const gko::LinOp* BlockMatrix::block_at(size_type i, size_type j) const
{
    return gko::as<gko::BlockOperator>(linop().get())->block_at(i, j);
}

/// Builds the linear combination ``Σ coeff_k * op_k`` as a single operator.
///
/// The result is lazy: it stores the operands (and a tiny scalar per coefficient)
/// and forms no combined matrix. The cost is paid at each ``apply`` as one
/// sub-apply per term.
///
/// :param ctx: Context the result lives on and registers with.
/// :param name: Name reported for the combination in timing output.
/// :param terms: ``(coefficient, operator)`` pairs; all operators must share a size.
/// :tparam T: Value type of the coefficients.
///
/// Operands are taken as ``shared_ptr<const gko::LinOp>``; any wrapper converts to
/// that implicitly, so passing a ``SparseMatrix`` (etc.) narrows straight to the
/// underlying operator with no copy of the handle and no object slicing.
template<Scalar T = double>
OperatorHandle combination(Context& ctx, std::string name,
                           std::vector<std::pair<T, std::shared_ptr<const gko::LinOp>>> terms);

/// Builds the operator product ``op0 * op1 * …`` as a single operator.
///
/// The product is applied right-to-left (``op_last`` first) and is lazy: no
/// product matrix is formed, so each ``apply`` chains the sub-applies through
/// cached intermediate vectors.
///
/// :param ctx: Context the result lives on and registers with.
/// :param name: Name reported for the composition in timing output.
/// :param ops: The operators to compose, left to right.
/// :tparam T: Value type of the operands.
///
/// Operands are taken as ``shared_ptr<const gko::LinOp>``; any wrapper converts to
/// that implicitly, so passing a ``SparseMatrix`` (etc.) narrows straight to the
/// underlying operator with no copy of the handle and no object slicing.
template<Scalar T = double>
OperatorHandle composition(Context& ctx, std::string name,
                           std::initializer_list<std::shared_ptr<const gko::LinOp>> ops);

template<Scalar T>
OperatorHandle combination(Context& ctx, std::string name,
                           std::vector<std::pair<T, std::shared_ptr<const gko::LinOp>>> terms)
{
    auto exec = ctx.executor();
    std::vector<std::shared_ptr<const gko::LinOp>> coefficients;
    std::vector<std::shared_ptr<const gko::LinOp>> operators;
    coefficients.reserve(terms.size());
    operators.reserve(terms.size());
    for (auto& [coefficient, op] : terms) {
        auto scalar = gko::matrix::Dense<T>::create(exec, gko::dim<2>{1, 1});
        scalar->fill(coefficient);
        coefficients.push_back(gko::share(std::move(scalar)));
        operators.push_back(op);
    }
    auto combined =
        gko::Combination<T>::create(coefficients.begin(), coefficients.end(), operators.begin(), operators.end());
    return make_operator_handle(ctx, std::move(name), gko::share(std::move(combined)));
}

template<Scalar T> OperatorHandle composition(Context& ctx, std::string name, std::initializer_list<OperatorHandle> ops)
{
    std::vector<std::shared_ptr<const gko::LinOp>> operators;
    operators.reserve(ops.size());
    for (const auto& op : ops) {
        operators.push_back(op);
    }
    auto composed = gko::Composition<T>::create(operators.begin(), operators.end());
    return make_operator_handle(ctx, std::move(name), gko::share(std::move(composed)));
}

/// The operator sum ``a + b``, as a lazy combination named ``"a+b"``.
///
/// Context is taken from the left operand. Like all the DSL operators this is
/// O(1) to build and recomputes both sub-applies on every apply, so for an
/// operator hammered in a hot Krylov loop prefer a materialized matrix; for
/// assembly and occasional operators it is ideal.
///
/// :relates: OperatorHandle
inline OperatorHandle operator+(const OperatorHandle& a, const OperatorHandle& b)
{
    return combination<double>(a.context(), a.name() + "+" + b.name() , {{1.0, a}, {1.0, b}});
}

/// The operator difference ``a - b``, as a lazy combination named ``"a-b"``.
///
/// :relates: OperatorHandle
inline OperatorHandle operator-(const OperatorHandle& a, const OperatorHandle& b)
{
    return combination<double>(a.context(), a.name() + "-" + b.name() , {{1.0, a}, {-1.0, b}});
}

/// The operator product ``a * b`` (apply ``b`` then ``a``), as a lazy composition named ``"a*b"``.
///
/// No product matrix is formed; the apply chains ``a(b(x))``.
///
/// :relates: OperatorHandle
inline OperatorHandle operator*(const OperatorHandle& a, const OperatorHandle& b)
{
    return composition<double>(a.context(), a.name() + "*" + b.name() , {a, b});
}

/// The scaled operator ``scalar * a``, as a lazy combination named ``"scalar*a"``.
///
/// :tparam T: Value type of the scalar.
/// :relates: OperatorHandle
template<Scalar T> OperatorHandle operator*(T scalar, const OperatorHandle& a)
{
    return combination<T>(a.context(), std::to_string(scalar) + "*" + a.name() , {{scalar, a}});
}

} // namespace miscibility::instrument
