#define _USE_MATH_DEFINES
#include <random>
//#include <Eigen/Dense>
#include <se3.hpp>
#include <sim3.hpp>
#include <sophus.hpp>

#include <iostream>
#include <math.h>

#include "gtest/gtest.h"


using namespace Eigen;
using namespace Sophus;
using namespace std;


template < class T >
Matrix<T, 3, 3> skew_sym(const Matrix<T, 3, 1>& vec_) {
	Matrix<T, 3, 3> out = Matrix<T, 3, 3>::Zero();
	out(0, 1) = -vec_(2);	out(0, 2) =  vec_(1);
	out(1, 0) =  vec_(2);	out(1, 2) = -vec_(0);
	out(2, 0) = -vec_(1);	out(2, 1) =  vec_(0);
	return out;
}

template < class T >
Matrix<T, 9, 3> dR_by_dv(const Matrix<T, 3, 1>& rv_, float thr_) {
	//reference:
	//if || rv_ ||^2 > thr_:
	//       use Eqn (2) in "gvnn : Neural Network Library for Geometric Computer Vision"
	//else
	//       use SO(3) generators which is the 1st-order approximation of the exponential map
	//       reference: Eqn (2.14) to (2.16) in "Lie groups, Lie algebras, projective geometry and optimization for 3D Geometry, Engineering and Computer Vision"
	
	Matrix<T, 3, 3> R = SO3Group<float>::exp(rv_).matrix();
	Matrix<T, 3, 3> ss_rv = SO3Group<T>::hat(rv_); //ss_rv the skew symmetric matrix of rv_ 
	Matrix<T, 3, 3> I = Matrix<T, 3, 3>::Identity(); 
	T n = rv_.squaredNorm(); 
	Matrix<T, 3, 3> dR;
	Matrix<T, 9, 3> out;
	for (int i = 0; i < 3; i++)
	{
		Matrix<T, 3, 1> e_i = I.col(i);
		if (n > thr_) {
			Matrix<T, 3, 1> tmp = ss_rv *  (I - R) * e_i; // rv_ x ( I -R ) * e_i 
			dR = (rv_(i) * ss_rv + SO3Group<T>::hat(tmp))*R / n;
		}
		else {
			dR = SO3Group<T>::hat(e_i);
		}
		Matrix<T, -1, 1> r(Map<Matrix<T, -1, 1>>(dR.data(), 9));
		out.col(i) = r;
	}
	return out;
}

template < class T >
Matrix<T, 9, 3> dR_by_dv_numerical(const Matrix<T, 3, 1>& rv_, float thr_) {
	//reference:
	Matrix<T, 9, 3> out;
	out.setZero();

	T delta = 0.0001f;
	Matrix<T, 3, 1> rv0, rv1;
	for (int i = 0; i < 3; i++)
	{
		rv0 = rv_;
		rv1 = rv_;
		rv0(i) = rv_(i) - delta;
		Matrix<T, 3, 3> R1 = SO3Group<T>::exp(rv0).matrix();
		rv1(i) = rv_(i) + delta;
		Matrix<T, 3, 3> R2 = SO3Group<T>::exp(rv1).matrix();
		Matrix<T, 3, 3> dR = 0.5*(R2 - R1) / delta;
		Matrix<T, -1, 1> r(Map<Matrix<T, -1, 1>>(dR.data(), 9));
		out.col(i) = r;
	}

	return out;
}

template < class T >
Matrix<T, 3, 3> dVu_by_dv(
	const Matrix<T, 3, 1> & omega, 
	const Matrix<T, 3, 1> & u) {
	//calc SO3, SE3, V, Omega = [w]x, and Omega_sq = [w]x^2
	T theta;
	const SO3Group<T> & so3 = SO3Group<T>::expAndTheta(omega, &theta);
	const Matrix<T, 3, 3> & Omega = SO3Group<T>::hat(omega);
	const Matrix<T, 3, 3> & Omega_sq = Omega*Omega;
	Matrix<T, 3, 3> V, Id;
	Id = Matrix<T, 3, 3>::Identity();
	T l_ct, st, inv_theta, inv_theta_sq, inv_theta_3;
	inv_theta = static_cast<T>(1) / theta;
	if (theta < SophusConstants<T>::epsilon()) {
		V = so3.matrix();
		//Note: That is an accurate expansion!
	}
	else {
		inv_theta_sq = inv_theta * inv_theta;
		inv_theta_3 = inv_theta_sq * inv_theta;

		l_ct = static_cast<T>(1) - std::cos(theta);
		st = std::sin(theta);
		V = (Id + l_ct * inv_theta_sq * Omega
			+ (theta - st) * inv_theta_3 * Omega_sq);
	}

	Matrix<T, 3, 3 > out;
	return out;
}

template < class T >
Matrix<T, 3, 3> calc_V(	const Matrix<T, 3, 1> & omega) {
	//calc SO3, SE3, V, Omega = [w]x, and Omega_sq = [w]x^2
	T theta;
	const SO3Group<T> & so3 = SO3Group<T>::expAndTheta(omega, &theta);
	const Matrix<T, 3, 3> & Omega = SO3Group<T>::hat(omega);
	const Matrix<T, 3, 3> & Omega_sq = Omega*Omega;
	Matrix<T, 3, 3> V, Id;
	Id = Matrix<T, 3, 3>::Identity();
	T l_ct, st, inv_theta, inv_theta_sq, inv_theta_3;
	inv_theta = static_cast<T>(1) / theta;
	if (theta < SophusConstants<T>::epsilon()) {
		V = so3.matrix();
		//Note: That is an accurate expansion!
	}
	else {
		inv_theta_sq = inv_theta * inv_theta;
		inv_theta_3 = inv_theta_sq * inv_theta;

		l_ct = static_cast<T>(1) - std::cos(theta);
		st = std::sin(theta);
		V = (Id + l_ct * inv_theta_sq * Omega
			+ (theta - st) * inv_theta_3 * Omega_sq);
	}
	return V;
}

template < class T >
Matrix<T, 3, 3> dVu_by_dv_numerical(
	const Matrix<T, 3, 1> & omega_,
	const Matrix<T, 3, 1> & u_) {
	Matrix<T, 3, 3> out;
	Matrix<T, 3, 1> t0, t1;

	T delta = 0.0001f;
	for (int i = 0; i < 3; i++)
	{
		t0 = omega_;
		t1 = omega_;
		t0(i) = omega_(i) - delta;
		Matrix<T, 3, 3> V0 = calc_V(t0);
		t1(i) = omega_(i) + delta;
		Matrix<T, 3, 3> V1 = calc_V(t1);
		out.col(i) = .5/delta*(V1 - V0)*u_;
	}
	return out;
}


template < class T >
Matrix<T, 12, 6> dse3_by_dv(const Matrix<T, 6, 1>& tangent_, float thr_) {
	//reference:
	Matrix<T, 12, 6> out; 
	out.setZero();
	//cout << out << endl;

	//calc SO3, SE3, V, Omega = [w]x, and Omega_sq = [w]x^2
	const Matrix<T, 3, 1> & omega = tangent_.template tail<3>();

	T theta;
	const SO3Group<T> & so3 = SO3Group<T>::expAndTheta(omega, &theta);
	const Matrix<T, 3, 3> & Omega = SO3Group<T>::hat(omega);
	const Matrix<T, 3, 3> & Omega_sq = Omega*Omega;
	Matrix<T, 3, 3> V, Id;
	Id = Matrix<T, 3, 3>::Identity();
	T l_ct, st, inv_theta, inv_theta_sq, inv_theta_3;
	inv_theta = static_cast<T>(1) / theta;
	if (theta < SophusConstants<T>::epsilon()) {
		V = so3.matrix();
		//Note: That is an accurate expansion!
	}
	else {
		inv_theta_sq = inv_theta * inv_theta;
		inv_theta_3 = inv_theta_sq * inv_theta;

		l_ct = static_cast<T>(1) - std::cos(theta);
		st = std::sin(theta);
		V = ( Id + l_ct * inv_theta_sq * Omega
				 + ( theta - st ) * inv_theta_3 * Omega_sq );
	}

	//Matrix<T, 3, 1> Vu = V*tangent_.template head<3>();

	// get dso2_by_omega
	Matrix<T, 9, 3> dso3 = dR_by_dv(omega, thr_);
	out.block(0, 3, 9, 3) = dso3;
	//cout << out << endl;

	Matrix<T, 3, 3> dVu;
	Matrix<T, 3, 1> normalised_om = omega * inv_theta;
	//calc dV_by_omega
	if (theta < SophusConstants<T>::epsilon()) {
		dVu.setZero();
	}
	else {
		Matrix<T, 3, 1> u = tangent_.template head<3>();
		if(false){
			dVu = dVu_by_dv_numerical(omega, u);
		}
		else{
			Matrix<T, 3, 3> dV;
			Matrix<T, 3, 1> e_i;
			
			for (int i = 0; i < 3; i++) {
				e_i = Id.col(i);
				if (true)
				{
					dV = (st * inv_theta_sq - 2 * l_ct * inv_theta_3) * normalised_om(i) * Omega + l_ct * inv_theta_sq *skew_sym(e_i) +
						(l_ct * inv_theta_3 - 3 * (theta - st) * inv_theta_sq * inv_theta_sq) * normalised_om(i) * Omega_sq +
						(theta - st) * inv_theta_3 * (e_i * omega.transpose() + omega * e_i.transpose() -2*omega(i)*Id );
				}
				else {
					dV = (-2 * (l_ct)+theta*st)*inv_theta_3 * normalised_om(i) * Omega + l_ct * inv_theta_sq * skew_sym(e_i) +
						(3 * (st - theta) + theta*l_ct)* inv_theta_sq*inv_theta_sq * normalised_om(i) * Omega_sq +
						(theta - st) * inv_theta_3 * (e_i * omega.transpose() + omega * e_i.transpose()-2*omega(i)*Id);
				}
				
				dVu.col(i) = dV * tangent_.template head<3>();
			}
		}
	}
	out.block(9, 0, 3, 3) = V;

	out.block(9, 3,3, 3) = dVu;
	//cout << out << endl;
	return out;
}

template < class T >
Matrix<T, 12, 6> dse3_by_dv_numerical(const Matrix<T, 6, 1>& tangent_, float thr_) {
	//reference:
	Matrix<T, 12, 6> out;
	out.setZero();

	T delta = 0.0001f;
	Matrix<T, 6, 1> t0,t1;
	for (int i = 0; i < 6; i++)
	{
		t0 = tangent_;
		t1 = tangent_;
		t0(i) = tangent_(i) - delta;
		Matrix<T, 4, 4> T1 = SE3Group<T>::exp(t0).matrix();
		t1(i) = tangent_(i) + delta;
		Matrix<T, 4, 4> T2 = SE3Group<T>::exp(t1).matrix();
		Matrix<T, 4, 4> dT_by_v = 0.5*(T2 - T1) / delta;
		for (int r = 0; r < 3; r++) {
			for (int c = 0; c < 3; c++) {
				out(c * 3 + r, i) = dT_by_v(r, c);
			}
			out(9 + r,i) = dT_by_v(r, 3);
		}
	}

	return out;
}

template < class T >
Matrix<T, 12, 7> dsim3_by_dv(const Matrix<T, 7, 1>& tangent_, float thr_) {
	//reference:
	Matrix<T, 12, 7> out;
	out.setZero();

	T delta = 0.0001f;
	Matrix<T, 7, 1> t0, t1;
	for (int i = 0; i < 7; i++)
	{
		t0 = tangent_;
		t1 = tangent_;
		t0(i) = tangent_(i) - delta;
		Matrix<T, 4, 4> T1 = Sim3Group<T>::exp(t0).matrix();
		t1(i) = tangent_(i) + delta;
		Matrix<T, 4, 4> T2 = Sim3Group<T>::exp(t1).matrix();
		Matrix<T, 4, 4> dT_by_v = 0.5*(T2 - T1) / delta;
		for (int r = 0; r < 3; r++) {
			for (int c = 0; c < 3; c++) {
				out(c * 3 + r, i) = dT_by_v(r, c);
			}
			out(9 + r, i) = dT_by_v(r, 3);
		}
	}

	return out;
}

template < class T >
Matrix<T, 12, 7> dsim3_by_dv_numerical(const Matrix<T, 7, 1>& tangent_, float thr_) {
	//reference:
	Matrix<T, 12, 7> out;
	out.setZero();

	T delta = 0.0001f;
	Matrix<T, 7, 1> t0, t1;
	for (int i = 0; i < 7; i++)
	{
		t0 = tangent_;
		t1 = tangent_;
		t0(i) = tangent_(i) - delta;
		Matrix<T, 4, 4> T1 = Sim3Group<T>::exp(t0).matrix();
		t1(i) = tangent_(i) + delta;
		Matrix<T, 4, 4> T2 = Sim3Group<T>::exp(t1).matrix();
		Matrix<T, 4, 4> dT_by_v = 0.5*(T2 - T1) / delta;
		for (int r = 0; r < 3; r++) {
			for (int c = 0; c < 3; c++) {
				out(c * 3 + r, i) = dT_by_v(r, c);
			}
			out(9 + r, i) = dT_by_v(r, 3);
		}
	}

	return out;
}

