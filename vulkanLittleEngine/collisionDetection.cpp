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

		// for cleaner code
		typedef thrust::tuple<bool, massElement, massElement, glm::vec3> collisionTuple;
		typedef thrust::tuple<massElement, massElement, glm::vec3> collisionPair;

		// detect collision between two particles
		struct detectCollisionParticle_Particle {
			const massElement p;

			detectCollisionParticle_Particle(massElement _p) : p{ _p } {}

			__host__ __device__ collisionTuple operator()(massElement p1) {
				// pi.radus = pj.radius = r for all i, j
				bool collided = glm::distance(p.particle.center, p1.particle.center) <= 2 * p.particle.radius;
				// average point
				glm::vec3 avg = 0.5f * (p.particle.center + p1.particle.center);
				return thrust::make_tuple<collisionTuple>(collided, p, p1, avg);
			}
		};

		// operator for summing over array to find number of true values and false ones
		struct isTrue {
			__host__ __device__ bool operator()(collisionTuple a) {
				return thrust::get<0>(a);
			}
		};

		// get collision pair from tuple
		struct getCollisionPair {
			__host__ __device__ collisionPair operator()(collisionTuple a) {
				return thrust::make_tuple(thrust::get<1>(a), thrust::get<2>(a), thrust::get<3>(a));
			}
		};

		/*
		calculate the contact point + add the massElements
		*/

		thrust::device_vector<collisionTuple> MLPE_RBP_COLLISION_DETECTOR::P_O_checkCollisionPoints(
			massElement m,
			mlpe_rbp_RigidBodyDynamicsInfo& OuterObjectInfo) {

			std::vector<massElement> OuterObjectParticles = OuterObjectInfo.massDistribution.massElements;
			thrust::device_vector<massElement> OOP_device = GeneralUsage::mlpe_gu_copyVector(OuterObjectParticles);
			// get boolean product for each particle
			thrust::device_vector<collisionTuple> res(OuterObjectParticles.size());
			thrust::transform(
				thrust::device,
				OOP_device.begin(),
				OOP_device.end(),
				res.begin(),
				detectCollisionParticle_Particle(m));
			return res;
		}

		// TODO : check when there are no points of collision
		thrust::device_vector<collisionPair> MLPE_RBP_COLLISION_DETECTOR::detectCollisionObject_Object(
			mlpe_rbp_RigidBodyDynamicsInfo OuterObjectInfo,
			mlpe_rbp_RigidBodyDynamicsInfo ObjectInfo) {

			// maximum number of collision points available
			int MAX_SIZE = ObjectInfo.massDistribution.massElements.size() * 
						   OuterObjectInfo.massDistribution.massElements.size();
			// initial place in collision_Pts to copy to
			int offset = 0;
			// collision
			thrust::device_vector<collisionTuple> collision_Pairs(MAX_SIZE);
			// can be more parallelized
			for (auto m : ObjectInfo.massDistribution.massElements) {
				thrust::device_vector<collisionTuple> PTODR = P_O_checkCollisionPoints(m, OuterObjectInfo);
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
			thrust::device_vector<collisionPair> collision_Pts(offset);
			// copy points into a "cleaner" array
			thrust::transform(
				thrust::device,
				collision_Pairs.begin(),
				collision_Pairs.begin() + offset,
				collision_Pts.begin(),
				getCollisionPair());
			return collision_Pts;
		}
	}
}