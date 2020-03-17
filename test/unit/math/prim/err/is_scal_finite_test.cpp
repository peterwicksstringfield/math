#include <stan/math/prim.hpp>
#include <gtest/gtest.h>
#include <limits>

using stan::math::is_scal_finite;

TEST(ErrorHandlingScalar, isScalFinite) {
  double x = 0;
  EXPECT_TRUE(is_scal_finite(x));

  x = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(is_scal_finite(x));

  x = -std::numeric_limits<double>::infinity();
  EXPECT_FALSE(is_scal_finite(x));

  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(is_scal_finite(x));
}

TEST(ErrorHandlingScalar, isScalFinite_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(is_scal_finite(nan));
}

TEST(ErrorHandlingScalar, isScalFinite_Matrix) {
  Eigen::Matrix<double, Eigen::Dynamic, 1> x;

  x.resize(3);
  x << -1, 0, 1;
  EXPECT_TRUE(is_scal_finite(x));

  x.resize(3);
  x << -1, 0, std::numeric_limits<double>::infinity();
  EXPECT_FALSE(is_scal_finite(x));

  x.resize(3);
  x << -1, 0, -std::numeric_limits<double>::infinity();
  EXPECT_FALSE(is_scal_finite(x));

  x.resize(3);
  x << -1, 0, std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(is_scal_finite(x));
}

TEST(ErrorHandlingScalar, isScalFinite_Matrix_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::Matrix<double, Eigen::Dynamic, 1> x_mat(3);
  x_mat << nan, 0, 1;
  EXPECT_FALSE(is_scal_finite(x_mat));

  x_mat << 1, nan, 1;
  EXPECT_FALSE(is_scal_finite(x_mat));

  x_mat << 1, 0, nan;
  EXPECT_FALSE(is_scal_finite(x_mat));
}
