#ifndef STAN_MATH_PRIM_META_IS_VECTOR_LIKE_HPP
#define STAN_MATH_PRIM_META_IS_VECTOR_LIKE_HPP

#include <stan/math/prim/meta/bool_constant.hpp>
#include <stan/math/prim/meta/is_eigen.hpp>
#include <stan/math/prim/meta/is_vector.hpp>
#include <type_traits>

namespace stan {

/** \ingroup type_trait
 * Template metaprogram indicates whether a type is vector_like.
 *
 * A type is vector_like if an instance can be accessed like a
 * vector, i.e. square brackets.
 *
 * Access is_vector_like::value for the result.
 *
 * Default behavior is to use the is_vector template metaprogram.
 *
 * @tparam T Type to test
 */
template <typename T>
struct is_vector_like
    : bool_constant<stan::is_vector<T>::value || std::is_pointer<T>::value
                    || is_eigen<T>::value || std::is_array<T>::value> {};

}  // namespace stan
#endif
