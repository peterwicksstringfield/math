#include <gtest/gtest.h>
#include <stan/math/prim/err/is_nonnegative.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <vector>

TEST(ErrorHandlingScalar, isNonnegative) {
  using stan::math::is_nonnegative;
  EXPECT_TRUE(is_nonnegative(-0.0));
  EXPECT_TRUE(is_nonnegative(0.0));
  EXPECT_TRUE(is_nonnegative((size_t)0));
  EXPECT_TRUE(is_nonnegative(0));
  EXPECT_TRUE(is_nonnegative(3.0));
  EXPECT_TRUE(is_nonnegative(std::vector<double>{3, 2}));
  EXPECT_TRUE(is_nonnegative(std::vector<int>{3, 2}));
  Eigen::MatrixXd m = Eigen::MatrixXd::Constant(3, 2, 0);
  EXPECT_TRUE(is_nonnegative(std::vector<Eigen::MatrixXd>{m, m}));
  EXPECT_TRUE(is_nonnegative(stan::math::INFTY));
  EXPECT_FALSE(is_nonnegative(-3.0));
  EXPECT_FALSE(is_nonnegative(std::vector<double>{3, -2}));
  Eigen::MatrixXd m2 = m;
  m2(1, 1) = -0.5;
  EXPECT_FALSE(is_nonnegative(std::vector<Eigen::MatrixXd>{m, m2, m}));
  EXPECT_FALSE(is_nonnegative(stan::math::NEGATIVE_INFTY));
  EXPECT_FALSE(is_nonnegative(stan::math::NOT_A_NUMBER));
}
