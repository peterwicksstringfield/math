#include <stan/math/prim.hpp>
#include <gtest/gtest.h>

TEST(ErrorHandlingScalar, isPositive) {
  using stan::math::is_positive;
  EXPECT_TRUE(is_positive(3.0));
  EXPECT_FALSE(is_positive(-3.0));
  EXPECT_FALSE(is_positive(-0.0));
  EXPECT_FALSE(is_positive(0.0));
  EXPECT_FALSE(is_positive(0));
}

TEST(ErrorHandlingScalar, isPositive_nan) {
  using stan::math::is_positive;
  EXPECT_FALSE(is_positive(stan::math::NEGATIVE_INFTY));
  EXPECT_TRUE(is_positive(stan::math::INFTY));
  EXPECT_FALSE(is_positive(stan::math::NOT_A_NUMBER));
}
