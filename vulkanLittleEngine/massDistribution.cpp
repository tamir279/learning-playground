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
#include <algorithm>
#include <stdexcept>
#include "physicsEngine.h"

/*
------------------- RIGID BODY PHYSICS -------------------
			DISTRIBUTING MASS OVER ALL PARTICLES
*/

namespace MLPE {
	namespace rbp {

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
			it = thrust::find_if(device_vec.begin(), device_vec.end(), greater_than_one());

			if (it != device_vec.end()) {
				throw::std::runtime_error("probability grater than 1");
			}
		}

		// needed to build a mass element - mass per particle + particle - from RigidBodyInfo
		void MLPE_RBP_massDistribution::distributeMassElements(mlpe_rbp_RigidBodyDynamicsInfo RigidBodyInfo) {
			checkVector(massDistrib.prob);
			thrust::device_vector<massElement> device_vec;
			thrust::transform(massDistrib.prob.begin(), massDistrib.prob.end(), device_vec.begin(), multiplyByConstant<float>(RigidBodyInfo.mass));
		}

		void MLPE_RBP_massDistribution::mass(mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo) {
			thrust::device_vector<massElement> massElems_d = copy_vec(massDistribution.massElements);
			float mass = thrust::reduce(massElems_d.begin(), massElems_d.end(), thrust_add_massElements<massElement>());
			RigidBodyInfo.mass = mass;
		}


		glm::vec3 MLPE_RBP_massDistribution::getCenterMass(mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo) {
			thrust::device_vector<massElement> massElems_d = copy_vec(massDistribution.massElements);
			glm::vec3 sum_vec = thrust::reduce(massElems_d.begin(), massElems_d.end(), thrust_add_Positions<massElement>());
			sum_vec *= 1 / RigidBodyInfo.mass;

			return sum_vec;
		}

		void MLPE_RBP_massDistribution::massElementsDistribution(mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo) {
			RigidBodyInfo.massDistribution = massDistribution;
		}
		
	}
}