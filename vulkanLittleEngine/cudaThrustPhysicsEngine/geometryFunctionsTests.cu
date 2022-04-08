#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>
#include <array>
#include "physicsEngine.cuh"

typedef std::vector<glm::vec3> ptVec;
typedef std::vector<uint32_t> indVec;
typedef std::vector<MLE::MLPE::rbp::polygon> polyVec;
typedef std::array<thrust::pair<glm::vec3, uint32_t>, 6> extrArr;

// for vertices
float yMid = glm::sqrt(2) / 2; float zMid = glm::sqrt(2) / 2;

// convert indices to polygons - helper
polyVec convertIndsToPolys(indVec indices, ptVec vertices) {
	polyVec res;
	MLE::MLPE::rbp::polygon poly;
	for (int i = 0; i < indices.size(); i += 3) {
		poly.polygon = thrust::make_tuple(vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]]);
		res.push_back(poly);
	}
	return res;
}

// tests for inner functions

// extremum type duplicate
template<typename T>
bool extremumType(T type) {
	return (type == "max") ? true : false;
}

// extremumOp struct and operator duplicate
template<typename T1, typename T2>
struct extremumOp {
	// define a constant that affects inside operator
	const T2 type;
	// initialize struct - constructor
	extremumOp(T2 _t) : type{ _t } {}

	__host__ __device__ bool operator()(T1 a, T1 b)const {
		return (type == "x") ? a.x < b.x : (type == "y") ? a.y < b.y : a.z < b.z;
	}
};

// extremum alogn axis duplicate
template<typename T1, typename T2>
thrust::pair<glm::vec3, uint32_t> extremumAlongAxis(T1 typeOfExtremum, T2 axis) {
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




int main() {
	/*
	check extremum points extraction
	*/
	MLE::MLPE::rbp::MLPE_RBP_RigidBodyGeometryInfo geometryInfo;
	// input vertices + indices
	ptVec v = { glm::vec3(0,0,1), glm::vec3(0, yMid, zMid), glm::vec3(0,1,0),
				glm::vec3(0, yMid, -zMid), glm::vec3(0,0,-1), glm::vec3(0, -yMid, -zMid),
				glm::vec3(0,-1,0), glm::vec3(0, -yMid, zMid), glm::vec3(1,0,0), glm::vec3(-1,0,0) };

	indVec inds = {0, 1, 8,
				   1, 2, 8,
				   2, 3, 8,
				   3, 4, 8,
				   4, 5, 8,
				   5, 6, 8,
				   6, 7, 8,
				   7, 0, 8,
				   0, 1, 9,
				   1, 2, 9,
				   2, 3, 9,
				   3, 4, 9,
				   4, 5, 9,
				   5, 6, 9,
				   6, 7, 9,
				   7, 0, 9};
	// get as an input to the geometry
	geometryInfo.vertices = v;
	geometryInfo.indices = inds;
	geometryInfo.objPolygons = convertIndsToPolys(inds, v);
	
	// test 1 - trigger the main function - get extremum points
	auto res1 = geometryInfo.getExtremumPoints();

	// test 2 - trigger the extremum along axis function - the main function depends on it
	auto res2_1 = extremumAlongAxis("max", "x");
	auto res2_2 = extremumAlongAxis("max", "y");
	auto res2_3 = extremumAlongAxis("max", "z");
	auto res2_4 = extremumAlongAxis("min", "x");
	auto res2_5 = extremumAlongAxis("min", "y");
	auto res2_6 = extremumAlongAxis("min", "z");

	if ((thrust::get<0>(res2_1)).x < (thrust::get<0>(res2_4)).x ||
		(thrust::get<0>(res2_2)).y < (thrust::get<0>(res2_5)).y ||
		(thrust::get<0>(res2_3)).z < (thrust::get<0>(res2_6)).z) std::cout << "test 2 failed"; return -1;

	// test 3 - trigger the extremumType function - there shouldn't be any problems with this one
	auto type1 = extremumType("min");
	auto type2 = extremumType("max");

	if (!type2 || type1)std::cout << "test 3 failed"; return -1;

	// test 4 - if tests 1 + 2 ran succesfully, compare values to ensure identical results
	extrArr res2Arr = { res2_4, res2_1, res2_5, res2_2, res2_6, res2_3 };
	for (int i = 0; i < 6; i++) {
		if (thrust::get<1>(res2Arr[i]) != thrust::get<1>(res1[i])) std::cout << "test 4 failed"; return -1;
	}

	return 0;
}