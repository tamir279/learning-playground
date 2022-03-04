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
			     FORCES APPLIED AT TIME T
*/

namespace MLPE {
	namespace rbp {

		// TODO : take into account the direction of motion during collision and output the forces and not contact points
		void MLPE_RBP_ForceStateDiagram::checkForCollisionForces(
			mlpe_rbp_RigidBodyDynamicsInfo bodyInfo,
			const std::vector<mlpe_rbp_RigidBodyDynamicsInfo> outerBodies) {

			// check for each body for contact points
			thrust::device_vector<glm::vec3> contactPts;
			for (auto body : outerBodies) {
				// the contact points
				thrust::device_vector<glm::vec3> pts = bodyInfo.detector.detectCollisionObject_Object(bodyInfo, body);
				// concatenate results
				contactPts.insert(contactPts.end(), pts.begin(), pts.end());
			}
			
			// copy vector
			CollisionForceDiagram = GeneralUsage::mlpe_gu_copyBackVector(contactPts);
		}

		// TODO for later - when user inputs are defined
		void MLPE_RBP_ForceStateDiagram::checkForUserForceInput() {

		}

		void MLPE_RBP_ForceStateDiagram::checkIfGravityEnabled(mlpe_rbp_RigidBodyDynamicsInfo bodyInfo) {
			thrust::device_vector<glm::vec3> gravityDistribution;
			// check if gravity is enabled in the simulation
			glm::vec3 gravity = (bodyInfo.GravityEnabled) ? bodyInfo.G * glm::vec3(0, 0, -1.0f) : glm::vec3(0);

			thrust::fill_n(
				thrust::device,
				gravityDistribution.begin(),
				bodyInfo.particleDecomposition.particleDecomposition.size(),
				gravity);

			// transfer the data into the built in variable
			GravitationForceDiagram = GeneralUsage::mlpe_gu_copyBackVector(gravityDistribution);
		}
	}
}