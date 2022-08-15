#include "body_physics.cuh"

const int cpu_limit = 1e5;
// data reading
void geometricData::readData(const std::string modelPath) {

}

void geometricData::polygonScan_cpu() {
	for (auto it = indices.begin(); it != indices.end(); it += 3) {
		cudaPoly polygon{};
		polygon.v1 = vertices[*it]; polygon.v2 = vertices[*it + 1];
		polygon.v3 = vertices[*it + 2];
		surfacePolygons.push_back(polygon);
	}
}

void geometricData::polygonScan_gpu() {
	// copy to device
	thrust::device_vector<int> dIndices(indices.begin(), indices.end());
	thrust::device_vector<float3> dVertices(vertices.begin(), vertices.end());
	// define iterators
	typedef thrust::device_vector<int>::iterator indIter;
	typedef thrust::device_vector<float3>::iterator vertexIter;
	// permute iterators to create vertices[indices[i]] for all i
	thrust::permutation_iterator<vertexIter, indIter> iter(dVertices.begin(), dIndices.begin());
	// create a zip iterator containing iter[i], iter[i+1], iter[i+2]
	// at first - create three strided iterators with a stride of 3
	auto stridedV1 = strided_iterator<thrust::permutation_iterator<vertexIter, indIter>>(iter, iter + indices.size() - 2, 3);
	auto stridedV2 = strided_iterator<thrust::permutation_iterator<vertexIter, indIter>>(iter + 1, iter + indices.size() - 1, 3);
	auto stridedV3 = strided_iterator<thrust::permutation_iterator<vertexIter, indIter>>(iter + 2, iter + indices.size(), 3);
	// secondly - create a zip iterator
	auto zip_iter = thrust::make_zip_iterator(thrust::make_tuple(stridedV1, stridedV2, stridedV3));
	// get each 3 vertices into a single polygon
	// thrust copy into a cudaPoly structure
}

// atomic function for data handling
void geometricData::getSurfacePolygons() {

	((int)vertices.size() > cpu_limit) ? polygonScan_gpu() : polygonScan_cpu();
}

void geometricData::fitBoundingBox() {

}

// copy from another geometry object
void geometricData::copyData(const geometricData& geometry) {

}