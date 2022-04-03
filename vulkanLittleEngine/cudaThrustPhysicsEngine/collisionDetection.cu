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
#include <thrust/remove.h>
#include <algorithm>
#include <stdexcept>
#include "physicsEngine.cuh"

/*
------------------- RIGID BODY PHYSICS -------------------
			DETECT COLLISIONS BETWEEN OBJECTS
*/

namespace MLE::MLPE {
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
				return thrust::make_tuple(collided, p, p1, avg);
			}
		};

		// operator for summing over array to find number of true values and false ones
		struct isTrue {
			__host__ __device__ bool operator()(collisionTuple a) {
				return !thrust::get<0>(a);
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

		std::vector<collisionTuple> MLPE_RBP_COLLISION_DETECTOR::P_O_checkCollisionPoints(
			massElement m,
			mlpe_rbp_RigidBodyDynamicsInfo& OuterObjectInfo) {

			std::vector<massElement> OuterObjectElems = OuterObjectInfo.massDistribution.massElements;
			// get boolean product for each particle
			std::vector<collisionTuple> res(OuterObjectElems.size());
			thrust::transform(
				thrust::device,
				thrust::device_pointer_cast(OuterObjectElems.data()),
				thrust::device_pointer_cast(OuterObjectElems.data()) + OuterObjectElems.size(),
				thrust::device_pointer_cast(res.data()),
				detectCollisionParticle_Particle(m));
			return res;
		}

		// TODO : check when there are no points of collision
		std::vector<collisionPair> MLPE_RBP_COLLISION_DETECTOR::detectCollisionObject_Object(
			mlpe_rbp_RigidBodyDynamicsInfo OuterObjectInfo,
			mlpe_rbp_RigidBodyDynamicsInfo ObjectInfo) {

			std::vector<collisionTuple> collision_Pairs;
			// can be more parallelized
			for (auto& m : ObjectInfo.massDistribution.massElements) {
				std::vector<collisionTuple> temp = P_O_checkCollisionPoints(m, OuterObjectInfo);
				std::vector<collisionTuple> PTODR;
				thrust::remove_copy_if(
					thrust::device,
					thrust::device_pointer_cast(temp.data()),
					thrust::device_pointer_cast(temp.data()) + temp.size(),
					thrust::device_pointer_cast(PTODR.data()),
					isTrue());

				// insert PTODR into the collision pairs
				collision_Pairs.insert(collision_Pairs.end(), PTODR.begin(), PTODR.end());
			}
			std::vector<collisionPair> collPairs(collision_Pairs.size());
			// get rid of the collision flag
			thrust::transform(
				thrust::device,
				thrust::device_pointer_cast(collision_Pairs.data()),
				thrust::device_pointer_cast(collision_Pairs.data()) + collision_Pairs.size(),
				thrust::device_pointer_cast(collPairs.data()),
				getCollisionPair());
			return collPairs;
		}
	}
}