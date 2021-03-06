#include <stan/math/prim/arr.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <limits>
#include <vector>

TEST(ErrorHandlingArr, checkNonzeroSizeMatrix) {
  using stan::math::check_nonzero_size;

  std::vector<double> a(4, 3);
  EXPECT_NO_THROW(stan::math::check_nonzero_size("checkNonzeroSize", "a", a));

  a.resize(2);
  EXPECT_NO_THROW(stan::math::check_nonzero_size("checkNonzeroSize", "a", a));

  a.resize(0);
  EXPECT_THROW_MSG(stan::math::check_nonzero_size("checkNonzeroSize", "a", a),
                   std::invalid_argument, "has size 0");
}

TEST(ErrorHandlingArr, checkNonzeroSizeMatrix_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  std::vector<double> a(4, nan);
  EXPECT_NO_THROW(stan::math::check_nonzero_size("checkNonzeroSize", "a", a));

  a.resize(2);
  EXPECT_NO_THROW(stan::math::check_nonzero_size("checkNonzeroSize", "a", a));

  a.resize(0);
  EXPECT_THROW_MSG(stan::math::check_nonzero_size("checkNonzeroSize", "a", a),
                   std::invalid_argument, "has size 0");
}
