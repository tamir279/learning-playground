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
#include "physicsEngine.cuh"
#include "exceptionHandling.h"

/*
------------------- RIGID BODY PHYSICS -------------------
			DISTRIBUTING MASS OVER ALL PARTICLES
*/

namespace MLE::MLPE {
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
			it = thrust::find_if(
				device_vec.begin(),
				device_vec.end(),
				greater_than_one());

			except::checkIfEnded(it, device_vec.end());
		}

		// needed to build a mass element - mass per particle + particle - from RigidBodyInfo - can be further optimized!
		void MLPE_RBP_massDistribution::distributeMassElements(mlpe_rbp_RigidBodyDynamicsInfo RigidBodyInfo) {
			// define transformed vectors
			thrust::device_vector<float> prob;
			std::vector<massElement> distribVec;
			// needed to transform massDistrib<float> to massDistrib2<massElement>
			thrust_wrapper_transform(
				true,
				massDistrib.prob.begin(),
				massDistrib.prob.end(),
				prob.begin(),
				multiplyByConstant<float>(RigidBodyInfo.mass));
			
			thrust_wrapper_transform(
				true,
				prob.begin(),
				prob.end(),
				RigidBodyInfo.particleDecomposition.particleDecomposition.begin(),
				RigidBodyInfo.particleDecomposition.particleDecomposition.end(),
				distribVec.begin(),
				mElementComb());
			massDistribution.massElements = distribVec;
		}

		void MLPE_RBP_massDistribution::Mass(mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo, float mass) {
			RigidBodyInfo.mass = mass;
		}


		void MLPE_RBP_massDistribution::getCenterMass(mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo) {
			glm::vec3 sum_vec = thrust_wrapper_reduce(
				true,
				massDistribution.massElements.begin(),
				massDistribution.massElements.end(),
				glm::vec3(0),
				thrust_add_Positions<massElement>());
				
			sum_vec *= 1 / RigidBodyInfo.mass;
			massDistribution.centerMass = sum_vec;
		}

		void MLPE_RBP_massDistribution::massElementsDistribution(mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo) {
			RigidBodyInfo.massDistribution = massDistribution;
		}

	}
}