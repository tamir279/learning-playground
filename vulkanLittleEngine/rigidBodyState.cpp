#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <array>

#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include<thrust/transform_scan.h>
#include <thrust/sequence.h>
#include <algorithm>
#include <stdexcept>
#include "physicsEngine.h"

/*
------------------- RIGID BODY PHYSICS -------------------
	    RIGID BODY PHYSICAL STATE AT SPECIFIC TIME T
*/

namespace MLPE {
	namespace rbp {	

		// calculate Icm 
		glm::mat3 calculateInitialInertiaTensor(
			glm::vec3 cm,
			std::vector<particle> pd,
			std::vector<massElement> me) {

			thrust::device_vector<glm::vec3> relativeDistance;
			thrust::device_vector<glm::mat3> kernelMatrix;
			// get relative distance to center of mass
			thrust::transform(thrust::device, pd.begin(), pd.end(), relativeDistance.begin(), minus(cm));
			// calculate kernel matrix for Inertia tensor - m((r^T*r)I - r*r^T)
			thrust::transform(thrust::device, relativeDistance.begin(), relativeDistance.end(), me.begin(), kernelMatrix.begin(), kernel());
			// get the inertia matrix - I = sum(m((r^T*r)I - r*r^T)) = sum(kernelMatrix)
			glm::mat3 I = thrust::reduce(thrust::device, kernelMatrix.begin(), kernelMatrix.end(), glm::mat3(0), GeneralUsage::Plus<glm::mat3>());
			return I;
		}

		// t = 0
		void MLPE_RBP_rigidBodyState::initializeState(mlpe_rbp_RigidBodyDynamicsInfo RigidBodyInfo) {
			// main parameters of a body state
			// r_cm
			thrust::get<0>(state_n.state) = RigidBodyInfo.massDistribution.centerMass;
			// q_obj
			MLPE_RBP_quaternion q(0, glm::vec3(0, 0, 1));
			q.ConvertToRotationQuaternionRepresentation();
			thrust::get<1>(state_n.state) = q;
			// P_cm
			thrust::get<2>(state_n.state) = glm::vec3(0, 0, 0);
			// L_cm
			thrust::get<3>(state_n.state) = glm::vec3(0, 0, 0);

			// auxilary parameters
			// inverse inertia tensor
			thrust::get<0>(state_n.auxilaryState) = 
				calculateInitialInertiaTensor(RigidBodyInfo.massDistribution.centerMass,
											  RigidBodyInfo.particleDecomposition.particleDecomposition,
											  RigidBodyInfo.massDistribution.massElements);
			// total force
			// torque
			// angular velocity
			thrust::get<3>(state_n.auxilaryState) = glm::vec3(0, 0, 1);
		}

		void MLPE_RBP_rigidBodyState::getPreviousState(mlpe_rbp_RigidBodyDynamicsInfo RigidBodyInfo) {
			state_n_m_1 = RigidBodyInfo.state;
			M = RigidBodyInfo.mass;
		}

		void MLPE_RBP_rigidBodyState::calculateCenterMass() {
			// r_n = r_(n-1) + DT*v_(n-1) = r_(n-1) + (DT/M)*P_(n-1)
			thrust::get<0>(state_n.state) = thrust::get<0>(state_n_m_1.state) + (dt / M) * thrust::get<2>(state_n_m_1.state);
		}

		// TODO : check if it is needed to notmalize and transform into rotation quaternion
		void MLPE_RBP_rigidBodyState::calculateRotationQuaternion() {
			// angular velocity quaternion from previous state
			MLPE_RBP_quaternion w(0, thrust::get<3>(state_n_m_1.auxilaryState));
			// previous rotation velocity
			MLPE_RBP_quaternion q_n = thrust::get<1>(state_n_m_1.state);
			// q_n+1 = q_n + DT/2 w_n*q_n
			thrust::get<1>(state_n.state) = q_n + (w * q_n) * (dt / 2.0f);
		}

		void MLPE_RBP_rigidBodyState::calculateLinearMomentum() {
			// P_n+1 = P_n + DT*F_n
			thrust::get<2>(state_n.state) = thrust::get<2>(state_n_m_1.state) + dt * thrust::get<1>(state_n_m_1.auxilaryState);
		}

		void MLPE_RBP_rigidBodyState::calculateAngularMomentum() {
			// L_n+1 = L_n + DT*Tau_n
			thrust::get<3>(state_n.state) = thrust::get<3>(state_n_m_1.state) + dt * thrust::get<2>(state_n_m_1.auxilaryState);
		}

		void MLPE_RBP_rigidBodyState::calculateTotalForce() {

		}

		void MLPE_RBP_rigidBodyState::calculateTorque() {

		}

		void MLPE_RBP_rigidBodyState::calculateInverseInertiaTensor() {

		}

		void MLPE_RBP_rigidBodyState::calculateAngularVelocity() {
			// w_n+1 = I_n+1 * L_n+1
			thrust::get<3>(state_n.auxilaryState) = thrust::get<0>(state_n.auxilaryState) * thrust::get<3>(state_n.state);
		}

		void MLPE_RBP_rigidBodyState::updateState(mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo) {
			// update state
			RigidBodyInfo.state = state_n;
			// update time step
			RigidBodyInfo.t_n = RigidBodyInfo.t_n + 1;
		}
	}
}