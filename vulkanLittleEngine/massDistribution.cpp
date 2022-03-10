#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <array>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <stdexcept>
#include "physicsEngine.h"

/*
------------------- RIGID BODY PHYSICS -------------------
			DISTRIBUTING MASS OVER ALL PARTICLES
*/

namespace MLPE {
	namespace rbp {

		// for cleaner code
		typedef mlpe_rbp_RigidBodyDynamicsInfo body;

		template<typename T>
		thrust::device_vector<T> MLPE_RBP_massDistribution::copy_vec(std::vector<T> vec) {
			thrust::host_vector<T> th_vec;
			thrust::copy(vec.begin(), vec.end(), th_vec.begin());
			thrust::device_vector<T> res = th_vec;

			return res;
		}

		template<typename T>
		void MLPE_RBP_massDistribution::checkVector(std::vector<T> p) {
			thrust::device_vector<T> device_vec = copy_vec(p);
			thrust::device_vector<T>::iterator it;
			it = thrust::find_if(
				thrust::device,
				device_vec.begin(),
				device_vec.end(),
				greater_than_one());

			if (it != device_vec.end()) {
				throw::std::runtime_error("probability greater than 1");
			}
		}

		// needed to build a mass element - mass per particle + particle - from RigidBodyInfo
		void MLPE_RBP_massDistribution::distributeMassElements(body RigidBodyInfo) {
			checkVector(massDistrib.prob);
			// define transformed vectors
			std::vector<float> prob;
			std::vector<massElement> distribVec;
			// needed to transform massDistrib<float> to massDistrib2<massElement>
			thrust::transform(
				thrust::device,
				massDistrib.prob.begin(),
				massDistrib.prob.end(),
				prob.begin(),
				multiplyByConstant<float>(RigidBodyInfo.mass));
			//                                massDistrib prob vector (mass of a particle)             particle info
			thrust::transform(
				thrust::device,
				prob.begin(),
				prob.end(),
				RigidBodyInfo.particleDecomposition.particleDecomposition.begin(), 
				distribVec.begin(),
				mElementComb());
			massDistribution.massElements = distribVec;
		}

		void MLPE_RBP_massDistribution::Mass(body& RigidBodyInfo, float mass) {
			RigidBodyInfo.mass = mass;
		}


		void MLPE_RBP_massDistribution::getCenterMass(body& RigidBodyInfo) {
			thrust::device_vector<massElement> massElems_d = copy_vec(massDistribution.massElements);
			glm::vec3 sum_vec = thrust::reduce(
				massElems_d.begin(),
				massElems_d.end(),
				glm::vec3(0),
				thrust_add_Positions<massElement>());
			sum_vec *= 1 / RigidBodyInfo.mass;
			massDistribution.centerMass = sum_vec;
		}

		void MLPE_RBP_massDistribution::massElementsDistribution(body& RigidBodyInfo) {
			RigidBodyInfo.massDistribution = massDistribution;
		}
		
	}
}