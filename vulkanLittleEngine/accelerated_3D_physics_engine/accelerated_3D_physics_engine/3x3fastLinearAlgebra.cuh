#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// row major
struct mat3_data {
	float3 row1;
	float3 row2;
	float3 row3;
};


class mat3 {
public:
	mat3_data data;

	mat3(mat3_data newData) {
		auto [r1, r2, r3] = newData;
		data = { r1, r2, r3 };
	}

	mat3() {
		data = { make_float3(0.0f, 0.0f, 0.0f),
				 make_float3(0.0f, 0.0f, 0.0f),
				 make_float3(0.0f, 0.0f, 0.0f) };
	}

	mat3(const mat3& matrix) {
		auto [r1, r2, r3] = matrix.data;
		data = { r1, r2, r3 };
	}

	// operators
	mat3& operator=(const mat3& matrix) {
		auto [r1, r2, r3] = matrix.data;
		data = { r1, r2, r3 };
		return *this;
	}

	mat3 operator+(const mat3& matrix) {
		mat3 res(data);
		auto [r1, r2, r3] = matrix.data;
		res.data = { make_float3(res.data.row1.x + r1.x, res.data.row1.y + r1.y, res.data.row1.z + r1.z),
					 make_float3(res.data.row2.x + r2.x, res.data.row2.y + r2.y, res.data.row2.z + r2.z),
					 make_float3(res.data.row3.x + r3.x, res.data.row3.y + r3.y, res.data.row3.z + r3.z) };
	}

	mat3 operator-(const mat3& matrix) {
		mat3 res(data);
		auto [r1, r2, r3] = matrix.data;
		res.data = { make_float3(res.data.row1.x - r1.x, res.data.row1.y - r1.y, res.data.row1.z - r1.z),
					 make_float3(res.data.row2.x - r2.x, res.data.row2.y - r2.y, res.data.row2.z - r2.z),
					 make_float3(res.data.row3.x - r3.x, res.data.row3.y + r3.y, res.data.row3.z - r3.z) };
	}

	mat3 operator*(const mat3& matrix) {
		auto m_d = matrix.data;
		convertFromRowMajorToColumnMajor(m_d);
		mat3 res({ make_float3(dot(data.row1, m_d.row1), dot(data.row1, m_d.row2), dot(data.row1, m_d.row3)),
				   make_float3(dot(data.row2, m_d.row1), dot(data.row2, m_d.row2), dot(data.row2, m_d.row3)),
			       make_float3(dot(data.row3, m_d.row1), dot(data.row3, m_d.row2), dot(data.row3, m_d.row3)) });
		
		return res;
	}

	float3 operator*(const float3& vec) {
		return make_float3(dot(data.row1, vec), dot(data.row2, vec), dot(data.row3, vec));
	}

	void inverse();

	mat3 Inverse();

private:

	void convertFromRowMajorToColumnMajor(mat3_data& row_matrix);
	void convertFromColumnMajorToRowMajor(mat3_data& col_matrix);
	float dot(float3 v1, float3 v2);
	float det(mat3_data matrix);
};


void mat3::convertFromRowMajorToColumnMajor(mat3_data& row_matrix) {
	auto [r1, r2, r3] = row_matrix;
	row_matrix = { make_float3(r1.x, r2.x, r3.x),
				   make_float3(r1.y, r2.y, r3.y),
				   make_float3(r1.z, r2.z, r3.z) };
}

void mat3::convertFromColumnMajorToRowMajor(mat3_data& col_matrix) {
	auto [r1, r2, r3] = col_matrix;
	col_matrix = { make_float3(r1.x, r2.x, r3.x),
				   make_float3(r1.y, r2.y, r3.y),
				   make_float3(r1.z, r2.z, r3.z) };
}

float mat3::dot(float3 v1, float3 v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

float mat3::det(mat3_data matrix) {

}