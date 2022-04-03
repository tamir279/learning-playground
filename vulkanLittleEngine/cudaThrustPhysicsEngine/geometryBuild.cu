#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <array>

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include<thrust/transform_scan.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sequence.h>
#include <algorithm>
#include <stdexcept>
#include "physicsEngine.cuh"

/*
------------------- RIGID BODY PHYSICS -------------------
	 GEOMETRY AND PARTICLE DECOMPOSITION OF AN OBJECT
*/

namespace MLE::MLPE {
	namespace rbp {

		// for cleaner code
		typedef thrust::device_vector<glm::vec3>::iterator glmIt;

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
			glmIt it; int N = (int)(vertices.size());
			// device pointer for device operations
			thrust::device_ptr<glm::vec3> devPtr = thrust::device_pointer_cast(vertices.data());
			// for all types of extremum
			if (extremumType(typeOfExtremum)) {
				// find maximum
				it = thrust::max_element(thrust::device, devPtr, devPtr + N, extremumOp<glm::vec3, std::string>(axis));
			}
			else {
				// find minimum
				it = thrust::min_element(thrust::device, devPtr, devPtr + N, extremumOp<glm::vec3, std::string>(axis));
			}

			// find index and value
			uint32_t extrIndex = static_cast<uint32_t>(it - vertices.begin());
			glm::vec3 extrVec = *it;

			// make pair
			return thrust::make_pair(extrVec, extrIndex);
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

		// get number of points at each axis - grid.
		template<typename T1, typename T2>
		thrust::tuple<T1, T1, T1> getGridSize(std::array<thrust::pair<T2, T1>, 6> extremumPts, float Dv) {
			T1 X_Buffer = static_cast<T1>(std::floor((thrust::get<0>(extremumPts[1]).x - thrust::get<0>(extremumPts[0]).x) / Dv));
			T1 Y_Buffer = static_cast<T1>(std::floor((thrust::get<0>(extremumPts[3]).y - thrust::get<0>(extremumPts[2]).y) / Dv));
			T1 Z_Buffer = static_cast<T1>(std::floor((thrust::get<0>(extremumPts[5]).z - thrust::get<0>(extremumPts[4]).z) / Dv));

			return thrust::make_tuple(X_Buffer, Y_Buffer, Z_Buffer);
		}

		// transfer grid size data to grid struct
		void MLPE_RBP_RIGIDBODY_GEOMETRY::get3DgridSize(thrust::tuple<uint32_t, uint32_t, uint32_t> gridSizes) {
			// get grid sizes
			grid.gridAxisSize = gridSizes;
			grid.gridSize = thrust::get<0>(gridSizes) *
				thrust::get<1>(gridSizes) *
				thrust::get<2>(gridSizes);
		}

		/*
		size of each array element is floor(max.x-min.x/2r) which is the size of array of cubes (or centroid points)
		in the x direction at specific y, z coordinates. therefore, there are N = L*floor(max.z-min.z/2r) elements,
		while L = floor(max.y-min.y/2r)

		min.y,min.z    y1, min.z      y2, min.z         max.y,min.z     min.y, z1        max.y, max.z

		   arr[0]        arr[1]          arr[2]            arr[L]         arr[L+1]          arr[N]
		 _ _ _ _ _ _    _ _ _ _ _ _    _ _ _ _ _ _        _ _ _ _ _ _    _ _ _ _ _ _       _ _ _ _ _ _
		|_|_|_|_|_|_|  |_|_|_|_|_|_|  |_|_|_|_|_|_| ...  |_|_|_|_|_|_|  |_|_|_|_|_|_| ... |_|_|_|_|_|_|

		the final size of grid is M = floor(max.x-min.x/2r)*N points,
		#floats = 3*floor(max.x-min.x/2r)*floor(max.y-min.y/2r)*floor(max.z-min.z/2r)
		NOTE: all of the elements are concetenated into one big array of size M

		/the points of the grid represent the centroid of a cube element of side length 2r/
		*/
		// WITHOUT TESTING, feels slow at the moment
		void MLPE_RBP_RIGIDBODY_GEOMETRY::get3Dgrid(MLPE_RBP_RigidBodyGeometryInfo GeometricInfo) {
			// differntials
			float Dv = 2 * GeometricInfo.r;
			// get boundries on each axis - using the vertices of an object
			std::array<thrust::pair<glm::vec3, uint32_t>, 6> extremumPts = GeometricInfo.getExtremumPoints();

			/*
			number of points at each axis : x_buffer, y_buffer, z_buffer
			*/
			thrust::tuple<uint32_t, uint32_t, uint32_t> gridSizes = getGridSize(extremumPts, Dv);

			// get size of grid and array element
			get3DgridSize(gridSizes);

			uint32_t xBuffer = thrust::get<0>(grid.gridAxisSize);
			uint32_t yBuffer = thrust::get<1>(grid.gridAxisSize);
			uint32_t zBuffer = thrust::get<2>(grid.gridAxisSize);

			// create 2D loop on y, z coordinates where thrust::tabulate will be applied on 1D array (x Direction)
			#pragma unroll(2)
			for (uint32_t z = 0; z < zBuffer; z++) {
				#pragma unroll(2)
				for (uint32_t y = 0; y < yBuffer; y++) {
					glm::vec3 initPos = glm::vec3(thrust::get<0>(extremumPts[0]).x + Dv / 2,
						thrust::get<0>(extremumPts[2]).y + Dv / 2 + Dv * (float)y,
						thrust::get<0>(extremumPts[4]).z + Dv / 2 + Dv * (float)z);
					// iterators
					thrust::device_ptr<glm::vec3> itBegin(grid.grid.data() + y * xBuffer + z * xBuffer * yBuffer);
					thrust::device_ptr<glm::vec3> itEnd(grid.grid.data() + xBuffer + y * xBuffer + z * xBuffer * yBuffer);
					//parallelized operations - create a sequence of vectors
					thrust::sequence(thrust::device, itBegin, itEnd, initPos, glm::vec3(Dv, 0.0, 0.0));
				}
			}
		}

		/*
		find if a centroid of a sub cube is inside the original object or not

		the algorithm:
		calculate the number of "windings" of the object around the point to check (each of the centroids)
		if the point is outside the number of windings will be 0, otherwise it will be a nonzero number (maybe negative)
		the calculation is done for each centroid individually
		*/

		// calculation of a poisition of a single point - boolean that returns "true" if the point is inside the object
		bool MLPE_RBP_RIGIDBODY_GEOMETRY::calcSignedngleForSpecificPoint(glm::vec3 p, std::vector<polygon> polygons) {
			// caclulate angle for each polygon
			std::vector<polygon> devPolys = polygons;
			// calculate total angle
			float res = thrust::reduce(
				thrust::device,
				thrust::make_transform_iterator(thrust::device_pointer_cast(devPolys.data()), solidAngle(p)),
				thrust::make_transform_iterator(thrust::device_pointer_cast(devPolys.data()) + devPolys.size(), solidAngle(p)),
				0.0f,
				thrust::plus<float>());
			return res >= 2.0f * glm::pi<float>();
		}

		// return a boolean array for each point  - "true" if the point is inside the object
		void MLPE_RBP_RIGIDBODY_GEOMETRY::isParticleInsideObject(MLPE_RBP_RigidBodyGeometryInfo GeometricInfo, std::vector<particle>& DC) {
			// get status of each centroid
			for (auto& Elem : grid.grid) {
				// if the centroid is inside the object, save it as a part of the particle decomposition of the object.
				if (calcSignedngleForSpecificPoint(Elem, GeometricInfo.objPolygons)) {
					// fill the particle data
					particle p{};
					p.radius = GeometricInfo.r;
					p.center = Elem;
					// save the particle
					DC.push_back(p);
				}
			}
		}

		void MLPE_RBP_RIGIDBODY_GEOMETRY::loadGeometry(
			MLPE_RBP_RigidBodyGeometryInfo GeometricInfo,
			mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo) {

			RigidBodyInfo.geometricInfo.vertices = GeometricInfo.vertices;
			RigidBodyInfo.geometricInfo.indices = GeometricInfo.indices;

			RigidBodyInfo.geometricInfo.objPolygons = GeometricInfo.objPolygons;
		}

		void MLPE_RBP_RIGIDBODY_GEOMETRY::decomposeGeomerty(MLPE_RBP_RigidBodyGeometryInfo GeometricInfo) {
			isParticleInsideObject(GeometricInfo, particleDecomposition.particleDecomposition);
		}

		void MLPE_RBP_RIGIDBODY_GEOMETRY::assignParticleDistribution(mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo) {
			RigidBodyInfo.particleDecomposition = particleDecomposition;
		}

	}
}