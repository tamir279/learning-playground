#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <array>

#include <thrust/extrema.h>
#include <algorithm>
#include <stdexcept>
#include "physicsEngine.h"

namespace MLPE {
	namespace rbp {

		// decide on type
		template<typename T>
		bool MLPE_RBP_RigidBodyGeometryInfo::extremumType(T type) {
			return (type == "max") ? true : false;
		}

		template<typename T>
		thrust::device_vector<T> MLPE_RBP_RigidBodyGeometryInfo::copy_vec(std::vector<T> vec) {
			thrust::host_vector<T> th_vec;
			thrust::copy(vec.begin(), vec.end(), th_vec.begin());
			thrust::device_vector<T> res = th_vec;

			return res;
		}

		// class geometric info - find general extremum
		template<typename T1, typename T2>
		thrust::pair<glm::vec3, uint32_t> MLPE_RBP_RigidBodyGeometryInfo::extremumAlongAxis(T1 typeOfExtremum, T2 axis) {
			thrust::device_vector<glm::vec3> d_vec = copy_vec(vertices);
			thrust::device_vector<glm::vec3>::iterator it;
			// for all types of extremum
			if (extremumType(typeOfExtremum)) {
				// find maximum
				it = thrust::max_element(d_vec.begin(), d_vec.end(), extremumOp<glm::vec3, std::string>(axis));
			}
			else {
				it = thrust::min_element(d_vec.begin(), d_vec.end(), extremumOp<glm::vec3, std::string>(axis));
			}

			// find index and value
			uint32_t extrIndex = (uint32_t)(it - d_vec.begin());
			glm::vec3 extrVec = *it;

			// make pair
			return thrust::make_pair<glm::vec3, uint32_t>(extrVec, extrIndex);
		}

		// find all extrema points in vertex grid
		std::array<thrust::pair<glm::vec3, uint32_t>, 6> MLPE_RBP_RigidBodyGeometryInfo::getExtremumPoints() {
			std::array<thrust::pair<glm::vec3, uint32_t>, 6> points;
			points[0] = extremumAlongAxis("min", "x");
			points[1] = extremumAlongAxis("max", "x");	
			points[2] = extremumAlongAxis("min", "y");
			points[3] = extremumAlongAxis("max", "y");
			points[4] = extremumAlongAxis("min", "z");
			points[5] = extremumAlongAxis("max", "z");

			return points;
		}

		// loads only model geometry - without textures, the model is loaded in the graphics("front end")
		void MLPE_RBP_RIGIDBODY_GEOMETRY::get3Dgrid(MLPE_RBP_RigidBodyGeometryInfo GeometricInfo) {

		}

		void MLPE_RBP_RIGIDBODY_GEOMETRY::loadGeometry(
			MLPE_RBP_RigidBodyGeometryInfo GeometricInfo,
			mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo) {

			RigidBodyInfo.geometricInfo.vertices = GeometricInfo.vertices;
			RigidBodyInfo.geometricInfo.indices = GeometricInfo.indices;
		}

		void MLPE_RBP_RIGIDBODY_GEOMETRY::decomposeGeomerty(MLPE_RBP_RigidBodyGeometryInfo GeometricInfo) {

		}


	}
}