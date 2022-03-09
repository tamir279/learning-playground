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
#include <thrust/count.h>
#include <algorithm>
#include <stdexcept>
#include "physicsEngine.h"

/*
------------------- RIGID BODY PHYSICS -------------------
	       	DETECT COLLISIONS BETWEEN OBJECTS
*/

namespace MLPE {
	namespace rbp {

		// TODO : output should be sts::vector<thrust::tuple<bool, massElement, massElement, glm::vec3>> res
		/*
		calculate the contact point + add the massElements
		*/
		thrust::device_vector<thrust::pair<bool, glm::vec3>> MLPE_RBP_COLLISION_DETECTOR::P_O_checkCollisionPoints(particle p, mlpe_rbp_RigidBodyDynamicsInfo& OuterObjectInfo) {
			std::vector<particle> OuterObjectParticles = OuterObjectInfo.particleDecomposition.particleDecomposition;
			thrust::device_vector<particle> OOP_device = GeneralUsage::mlpe_gu_copyVector(OuterObjectParticles);
			// get boolean product for each particle
			thrust::device_vector<thrust::pair<bool, glm::vec3>> res(OuterObjectParticles.size());
			thrust::transform(
				thrust::device,
				OOP_device.begin(),
				OOP_device.end(),
				res.begin(),
				detectCollisionParticle_Particle(p));
			return res;
		}

		// TODO : check when there are no points of collision
		thrust::device_vector<glm::vec3> MLPE_RBP_COLLISION_DETECTOR::detectCollisionObject_Object(
			mlpe_rbp_RigidBodyDynamicsInfo OuterObjectInfo, 
		    mlpe_rbp_RigidBodyDynamicsInfo ObjectInfo) {
			// maximum number of collision points available
			int MAX_SIZE = ObjectInfo.particleDecomposition.particleDecomposition.size() * 
						   OuterObjectInfo.particleDecomposition.particleDecomposition.size();
			// initial place in collision_Pts to copy to
			int offset = 0;
			// collision
			thrust::device_vector<thrust::pair<bool, glm::vec3>> collision_Pairs(MAX_SIZE);
			// can be more parallelized
			for (auto p : ObjectInfo.particleDecomposition.particleDecomposition) {
				thrust::device_vector<thrust::pair<bool, glm::vec3>> PTODR = P_O_checkCollisionPoints(p, OuterObjectInfo);
				thrust::copy_if(
					thrust::device,
					PTODR.begin(),
					PTODR.end(),
					collision_Pairs.begin() + offset,
					isTrue());

				offset += thrust::count_if(
					thrust::device,
					PTODR.begin(),
					PTODR.end(),
					isTrue());
			}
			// create a clean point array that containes all collision points
			thrust::device_vector<glm::vec3> collision_Pts(offset);
			// copy points into a "cleaner" array
			thrust::transform(
				thrust::device,
				collision_Pairs.begin(),
				collision_Pairs.begin() + offset,
				collision_Pts.begin(),
				secondArgument<bool, glm::vec3>());
			return collision_Pts;
		}
	}
}