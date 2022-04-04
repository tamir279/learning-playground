#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>
#include <cmath>

#include "physicsEngine.cuh"

/*
------------------- RIGID BODY PHYSICS -------------------
		  QUATERNION MATHEMATICS FOR 3D ROTATIONS
*/

namespace MLE::MLPE {
	namespace rbp {

		void MLPE_RBP_quaternion::operator+=(const MLPE_RBP_quaternion& q) {
			s += q.s;
			vector += q.vector;
		}

		MLPE_RBP_quaternion MLPE_RBP_quaternion::operator+(const MLPE_RBP_quaternion& q) {
			MLPE_RBP_quaternion q_res;
			q_res.s = s + q.s;
			q_res.vector = vector + q.vector;

			return q_res;
		}

		void MLPE_RBP_quaternion::operator-=(const MLPE_RBP_quaternion& q) {
			s -= q.s;
			vector -= q.vector;
		}

		MLPE_RBP_quaternion MLPE_RBP_quaternion::operator-(const MLPE_RBP_quaternion& q) {
			MLPE_RBP_quaternion q_res;
			q_res.s = s - q.s;
			q_res.vector = vector - q.vector;

			return q_res;
		}

		/*
		quat *= q <=> quat = quat * q

		quat = [s, vector], q = [s', vector']
		=> quat = [s*s' - dot(vector, vector'), s*vector' + s'*vector + cross(vector, vector')
		*/
		void MLPE_RBP_quaternion::operator*=(const MLPE_RBP_quaternion& q) {
			s = s * q.s - glm::dot(vector, q.vector);
			vector = q.vector * s + vector * q.s + glm::cross(vector, q.vector);
		}

		/*
		res = quat * q
		*/
		MLPE_RBP_quaternion MLPE_RBP_quaternion::operator*(const MLPE_RBP_quaternion& q) {
			MLPE_RBP_quaternion q_res;
			q_res.s = s * q.s - glm::dot(vector, q.vector);
			q_res.vector = q.vector * s + vector * q.s + glm::cross(vector, q.vector);

			return q_res;
		}

		// scalar multiplication
		void MLPE_RBP_quaternion::operator*=(const float scale) {
			s *= scale;
			vector *= scale;
		}

		MLPE_RBP_quaternion MLPE_RBP_quaternion::operator*(const float scale) {
			MLPE_RBP_quaternion q_res;
			q_res.s = s * scale;
			q_res.vector = vector * scale;

			return q_res;
		}

		/*
		a fast square root algorithm for accelerated (and mostly precise)
		approximation of the square root of a number
		*/
		float MLPE_RBP_quaternion::fastSquareRoot(float num) {
			// initial guess
			float x = 10.0;
			int iters = num > 35 ? 2 : 3;

			int i = 0;
			while (i < iters) {
				x = (x * x * x + 3 * num * x) / (3 * x * x + num);
				i++;
			}

			return x;
		}

		/*
		||q|| = square_root(s^2 + ||v||^2)
		*/
		float MLPE_RBP_quaternion::Norm() {
			float sq_norm = s * s + glm::dot(vector, vector);
			float norm = fastSquareRoot(sq_norm);

			return norm;
		}

		/*
		q' = q/||q||
		*/
		void MLPE_RBP_quaternion::Normalize() {

			if (!Norm()) {
				throw std::runtime_error("failed to Normalize quaternion : dividing by zero!");
			}

			float inverseNorm = 1 / Norm();

			// calculating components
			s *= inverseNorm;
			vector *= inverseNorm;
		}

		/*
		q* = [s, -v]
		*/
		void MLPE_RBP_quaternion::conjugate() {
			vector *= -1.0f;
		}

		MLPE_RBP_quaternion MLPE_RBP_quaternion::Conjugate() {
			MLPE_RBP_quaternion q_conjugate;
			q_conjugate.s = s;
			q_conjugate.vector = vector * (-1.0f);

			return q_conjugate;
		}

		/*
		q^-1 = q* /||q||^2
		*/
		MLPE_RBP_quaternion MLPE_RBP_quaternion::inverse() {
			MLPE_RBP_quaternion q_inverse = Conjugate();
			float norm = q_inverse.Norm();
			q_inverse.Normalize();
			q_inverse *= 1 / norm;

			return q_inverse;
		}

		// helper for converting degrees to radians
		float MLPE_RBP_quaternion::DegreesToRadians(float angle) {
			return angle * glm::pi<float>() / 180.0f;
		}

		/*
		v' = v/||v||
		q_unit = [cos(o/2), sin(o/2)v']
		*/
		void MLPE_RBP_quaternion::ConvertToRotationQuaternionRepresentation() {
			float Rangle = DegreesToRadians(s);
			s = std::cosf(Rangle / 2);
			vector = glm::normalize(vector) * std::sinf(Rangle / 2);
		}
	}
}