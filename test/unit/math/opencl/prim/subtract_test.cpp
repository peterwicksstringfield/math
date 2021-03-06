#ifdef STAN_OPENCL
#include <stan/math/prim/mat.hpp>
#include <stan/math/opencl/opencl.hpp>
#include <gtest/gtest.h>
#include <algorithm>

TEST(MathMatrixCL, subtract_v_exception_pass) {
  stan::math::vector_d d1, d2;

  d1.resize(3);
  d2.resize(3);
  stan::math::matrix_cl<double> d11(d1);
  stan::math::matrix_cl<double> d22(d2);
  stan::math::matrix_cl<double> d33(3, 1);
  EXPECT_NO_THROW(d33 = d11 - d22);
}

// TODO(Steve): This should probably throw expection?
// The dimensions do not match
TEST(MathMatrixCL, subtract_v_exception_pass_zero) {
  stan::math::vector_d d1, d2;
  d1.resize(0);
  d2.resize(0);
  stan::math::matrix_cl<double> d11(d1);
  stan::math::matrix_cl<double> d22(d2);
  stan::math::matrix_cl<double> d33(0, 1);
  EXPECT_NO_THROW(d33 = d11 - d22);
}

TEST(MathMatrixCL, subtract_v_exception_fail_zero) {
  stan::math::vector_d d1, d2;
  d1.resize(2);
  d2.resize(3);
  stan::math::matrix_cl<double> d11(d1);
  stan::math::matrix_cl<double> d22(d2);
  stan::math::matrix_cl<double> d33(3, 3);
  EXPECT_THROW(d33 = d11 - d22, std::invalid_argument);
}

TEST(MathMatrixCL, subtract_rv_exception_pass) {
  stan::math::row_vector_d d1, d2;

  d1.resize(3);
  d2.resize(3);
  stan::math::matrix_cl<double> d11(d1);
  stan::math::matrix_cl<double> d22(d2);
  stan::math::matrix_cl<double> d33(1, 3);
  EXPECT_NO_THROW(d33 = d11 - d22);
}

TEST(MathMatrixCL, subtract_rv_exception_pass_zero) {
  stan::math::row_vector_d d1, d2;
  d1.resize(0);
  d2.resize(0);
  stan::math::matrix_cl<double> d11(d1);
  stan::math::matrix_cl<double> d22(d2);
  stan::math::matrix_cl<double> d33(1, 0);
  EXPECT_NO_THROW(d33 = d11 - d22);
}

TEST(MathMatrixCL, subtract_rv_exception_fail_zero) {
  stan::math::row_vector_d d1, d2;
  d1.resize(2);
  d2.resize(3);
  stan::math::matrix_cl<double> d11(d1);
  stan::math::matrix_cl<double> d22(d2);
  stan::math::matrix_cl<double> d33(3, 1);
  EXPECT_THROW(d33 = d11 - d22, std::invalid_argument);
}

TEST(MathMatrixCL, subtract_m_exception) {
  stan::math::matrix_d d1, d2;
  d1.resize(2, 3);
  d2.resize(2, 3);
  stan::math::matrix_cl<double> d11(d1);
  stan::math::matrix_cl<double> d22(d2);
  stan::math::matrix_cl<double> d33(2, 3);
  EXPECT_NO_THROW(d33 = d11 - d22);
}

TEST(MathMatrixCL, subtract_m_exception_pass_zero) {
  stan::math::matrix_d d1, d2;
  d1.resize(0, 0);
  d2.resize(0, 0);
  stan::math::matrix_cl<double> d11(d1);
  stan::math::matrix_cl<double> d22(d2);
  stan::math::matrix_cl<double> d33(0, 0);
  EXPECT_NO_THROW(d33 = d11 - d22);
}

TEST(MathMatrixCL, subtract_m_exception_fail) {
  stan::math::matrix_d d1, d2;
  d1.resize(2, 3);
  d2.resize(3, 3);
  stan::math::matrix_cl<double> d11(d1);
  stan::math::matrix_cl<double> d22(d2);
  stan::math::matrix_cl<double> d33(3, 2);
  EXPECT_THROW(d33 = d11 - d22, std::invalid_argument);
}

TEST(MathMatrixCL, subtract_exception) {
  stan::math::vector_d v1(2);
  v1 << 1, 2;
  stan::math::vector_d v2(3);
  v2 << 10, 100, 1000;

  stan::math::row_vector_d rv1(2);
  v1 << 1, 2;
  stan::math::row_vector_d rv2(3);
  v2 << 10, 100, 1000;

  stan::math::matrix_d m1(2, 3);
  m1 << 1, 2, 3, 4, 5, 6;
  stan::math::matrix_d m2(3, 2);
  m2 << 10, 100, 1000, 0, -10, -12;

  using stan::math::matrix_cl;
  using stan::math::subtract;
  matrix_cl<double> v11(v1);
  matrix_cl<double> v22(v2);
  matrix_cl<double> v33(v1);
  matrix_cl<double> rv11(rv1);
  matrix_cl<double> rv22(rv2);
  matrix_cl<double> rv33(rv1);
  matrix_cl<double> m11(m1);
  matrix_cl<double> m22(m2);
  matrix_cl<double> m33(m1);
  EXPECT_THROW(v33 = v11 - v22, std::invalid_argument);
  EXPECT_THROW(rv33 = rv11 - rv22, std::invalid_argument);
  EXPECT_THROW(m33 = m11 - m22, std::invalid_argument);
}

TEST(MathMatrixCL, subtract_value_check) {
  stan::math::vector_d v1(3);
  v1 << 1, 2, 3;
  stan::math::vector_d v2(3);
  v2 << 10, 100, 1000;
  stan::math::vector_d v3(3);

  stan::math::row_vector_d rv1(3);
  rv1 << 1, 2, 3;
  stan::math::row_vector_d rv2(3);
  rv2 << 10, 100, 1000;
  stan::math::row_vector_d rv3(3);

  stan::math::matrix_d m1(3, 3);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  stan::math::matrix_d m2(3, 3);
  m2 << 10, 100, 1000, 0, -10, -12, 2, 4, 8;
  stan::math::matrix_d m3(3, 3);

  using stan::math::matrix_cl;
  using stan::math::subtract;
  matrix_cl<double> v11(v1);
  matrix_cl<double> v22(v2);
  matrix_cl<double> v33(3, 1);
  matrix_cl<double> rv11(rv1);
  matrix_cl<double> rv22(rv2);
  matrix_cl<double> rv33(1, 3);
  matrix_cl<double> m11(m1);
  matrix_cl<double> m22(m2);
  matrix_cl<double> m33(3, 3);

  EXPECT_NO_THROW(v33 = v11 - v22);
  EXPECT_NO_THROW(rv33 = rv11 - rv22);
  EXPECT_NO_THROW(m33 = m11 - m22);

  v3 = stan::math::from_matrix_cl(v33);
  EXPECT_EQ(-9, v3(0));
  EXPECT_EQ(-98, v3(1));
  EXPECT_EQ(-997, v3(2));

  rv3 = stan::math::from_matrix_cl(rv33);
  EXPECT_EQ(-9, rv3(0));
  EXPECT_EQ(-98, rv3(1));
  EXPECT_EQ(-997, rv3(2));

  m3 = stan::math::from_matrix_cl(m33);
  EXPECT_EQ(-9, m3(0, 0));
  EXPECT_EQ(-98, m3(0, 1));
  EXPECT_EQ(-997, m3(0, 2));
  EXPECT_EQ(4, m3(1, 0));
  EXPECT_EQ(15, m3(1, 1));
  EXPECT_EQ(18, m3(1, 2));
  EXPECT_EQ(5, m3(2, 0));
  EXPECT_EQ(4, m3(2, 1));
  EXPECT_EQ(1, m3(2, 2));
}
TEST(MathMatrixCL, subtract_tri_value_check) {
  Eigen::MatrixXd a(3, 3);
  a << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  Eigen::MatrixXd b = Eigen::MatrixXd::Ones(3, 3) * 3;
  stan::math::matrix_cl<double> a_cl(a);
  stan::math::matrix_cl<double> b_cl(b);
  stan::math::matrix_cl<double> c_cl(3, 3);
  Eigen::MatrixXd c(3, 3);

  a_cl.view(stan::math::matrix_cl_view::Lower);
  b_cl.view(stan::math::matrix_cl_view::Lower);
  c_cl = a_cl - b_cl;
  EXPECT_EQ(c_cl.view(), stan::math::matrix_cl_view::Lower);
  c = stan::math::from_matrix_cl(c_cl);
  EXPECT_EQ(-2, c(0, 0));
  EXPECT_EQ(1, c(1, 0));
  EXPECT_EQ(2, c(1, 1));
  EXPECT_EQ(4, c(2, 0));
  EXPECT_EQ(5, c(2, 1));
  EXPECT_EQ(6, c(2, 2));

  a_cl.view(stan::math::matrix_cl_view::Lower);
  b_cl.view(stan::math::matrix_cl_view::Upper);
  c_cl = a_cl - b_cl;
  EXPECT_EQ(c_cl.view(), stan::math::matrix_cl_view::Entire);
  c = stan::math::from_matrix_cl(c_cl);
  EXPECT_EQ(-2, c(0, 0));
  EXPECT_EQ(-3, c(0, 1));
  EXPECT_EQ(-3, c(0, 2));
  EXPECT_EQ(4, c(1, 0));
  EXPECT_EQ(2, c(1, 1));
  EXPECT_EQ(-3, c(1, 2));
  EXPECT_EQ(7, c(2, 0));
  EXPECT_EQ(8, c(2, 1));
  EXPECT_EQ(6, c(2, 2));

  a_cl.view(stan::math::matrix_cl_view::Upper);
  b_cl.view(stan::math::matrix_cl_view::Lower);
  c_cl = a_cl - b_cl;
  EXPECT_EQ(c_cl.view(), stan::math::matrix_cl_view::Entire);
  c = stan::math::from_matrix_cl(c_cl);
  EXPECT_EQ(-2, c(0, 0));
  EXPECT_EQ(2, c(0, 1));
  EXPECT_EQ(3, c(0, 2));
  EXPECT_EQ(-3, c(1, 0));
  EXPECT_EQ(2, c(1, 1));
  EXPECT_EQ(6, c(1, 2));
  EXPECT_EQ(-3, c(2, 0));
  EXPECT_EQ(-3, c(2, 1));
  EXPECT_EQ(6, c(2, 2));

  a_cl.view(stan::math::matrix_cl_view::Entire);
  b_cl.view(stan::math::matrix_cl_view::Lower);
  c_cl = a_cl - b_cl;
  EXPECT_EQ(c_cl.view(), stan::math::matrix_cl_view::Entire);
  c = stan::math::from_matrix_cl(c_cl);
  EXPECT_EQ(-2, c(0, 0));
  EXPECT_EQ(2, c(0, 1));
  EXPECT_EQ(3, c(0, 2));
  EXPECT_EQ(1, c(1, 0));
  EXPECT_EQ(2, c(1, 1));
  EXPECT_EQ(6, c(1, 2));
  EXPECT_EQ(4, c(2, 0));
  EXPECT_EQ(5, c(2, 1));
  EXPECT_EQ(6, c(2, 2));
}
#endif
