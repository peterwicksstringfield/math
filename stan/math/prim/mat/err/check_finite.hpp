#ifndef STAN_MATH_PRIM_MAT_ERR_CHECK_FINITE_HPP
#define STAN_MATH_PRIM_MAT_ERR_CHECK_FINITE_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/scal/err/throw_domain_error.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/mat/fun/value_of.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

namespace stan {
namespace math {

/*
 * Return <code>true</code> is the specified matrix is finite.
 * @tparams T scalar type of the matrix, requires class method
 *   <code>.size()</code>
 * @tparam R number of rows or Eigen::Dynamic
 * @tparam C number of columns or Eigen::Dynamic
 * @param function name of function (for error messages)
 * @param name variable name (for error messages)
 * @param y matrix to test
 * @return <code>true</code> if the matrix is finite
 **/
namespace internal {
template <typename T, int R, int C>
struct finite<Eigen::Matrix<T, R, C>, true> {
  static void check(const char* function, const char* name,
                    const Eigen::Matrix<T, R, C>& y) {
    if (!value_of(y).allFinite()) {
      for (int n = 0; n < y.size(); ++n) {
        if (!(boost::math::isfinite)(y(n))) {
          throw_domain_error_vec(function, name, y, n, "is ",
                                 ", but must be finite!");
        }
      }
    }
  }
};
}  // namespace internal
}  // namespace math
}  // namespace stan

#endif
