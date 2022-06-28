#include "quaternion_math.h"

// basic vector operations
// dot product
float dot(float3 v1, float3 v2){
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

// cross product
float3 cross(float3 v1, float3 v2){
	return make_float3(v1.y * v2.z - v2.y * v1.z, v2.x * v1.z - v1.y * v2.z, v1.x * v2.y - v2.x * v1.y);
}

// library functions

quaternion& quaternion::operator=(const quaternion& q){
	s = q.s;
	vector.x = q.vector.x; 
	vector.y = q.vector.y;
	vector.z = q.vector.z;
	return *this;
}

void quaternion::operator+=(const quaternion& q) {
	s += q.s;
	vector.x += q.vector.x; 
	vector.y += q.vector.y;
	vector.z += q.vector.z;
}

quaternion quaternion::operator+(const quaternion& q) {
	quaternion q_res;
	q_res += q;
	return q_res;
}

void quaternion::operator-=(const quaternion& q) {
	s -= q.s;
	vector.x -= q.vector.x; 
	vector.y -= q.vector.y;
	vector.z -= q.vector.z;
}

quaternion  quaternion::operator-(const quaternion& q) {
	quaternion q_res;
	q_res -= q;
	return q_res;
}

/*
quat *= q <=> quat = quat * q

quat = [s, vector], q = [s', vector']
=> quat = [s*s' - dot(vector, vector'), s*vector' + s'*vector + cross(vector, vector')
*/
void quaternion::operator*=(const quaternion& q) {
	s = s * q.s - dot(vector, q.vector);
	float3 cross_prod = cross(vector, q.vector);
	vector.x = q.vector.x * s + vector.x * q.s + cross_prod.x;
	vector.y = q.vector.y * s + vector.y * q.s + cross_prod.y;
	vector.z = q.vector.z * s + vector.z * q.s + cross_prod.z;
}

/*
res = quat * q
*/
quaternion quaternion::operator*(const quaternion& q) {
	quaternion q_res;
	q_res *= q;
	return q_res;
}

// scalar multiplication
void quaternion::operator*=(const float scale) {
	s *= scale;
	vector.x *= scale; vector.y *= scale; vector.z *= scale;
}

quaternion quaternion::operator*(const float scale) {
	quaternion q_res;
	q_res *= scale;
	return q_res;
}

// for multiplying scalars on the left...
quaternion operator*(const float scale, const quaternion& q) {
	quaternion q_res;
	q_res.s = q.s * scale;
	q_res.vector.x = q.vector.x * scale;
	q_res.vector.y = q.vector.y * scale;
	q_res.vector.z = q.vector.z * scale;
	return q_res;
}

/*
a fast square root algorithm for accelerated (and mostly precise)
approximation of the square root of a number
*/
float quaternion::fastSquareRoot(float num) {
	// initial guess
	float x = 10.0;
	int iters = num > 35 ? 2 : 3;

	int i = 0;
	while (i < iters) {
		x = (x * x * x + 3 * num * x) / (3 * x * x + num);
		i++;
	}

	return x;
}

/*
||q|| = square_root(s^2 + ||v||^2)
*/
float quaternion::Norm() {
	return fastSquareRoot(s * s + dot(vector, vector));
}

/*
q' = q/||q||
*/
void quaternion::Normalize() {

	if (!Norm()) {
		throw std::runtime_error("failed to Normalize quaternion : dividing by zero!");
	}

	// calculating components
	*this *= 1.0f / Norm();
}

/*
q* = [s, -v]
*/
void quaternion::conjugate() {
	vector.x *= -1.0f; vector.y *= -1.0f; vector.z *= -1.0f;
}

quaternion quaternion::Conjugate() {
	quaternion q_conjugate = *this;
	q_conjugate.conjugate();
	return q_conjugate;
}

/*
q^-1 = q* /||q||^2
*/
quaternion quaternion::inverse() {
	quaternion q_inverse = Conjugate();
	float norm = q_inverse.Norm();
	q_inverse.Normalize();
	q_inverse *= 1.0f / norm;
	return q_inverse;
}

// helper for converting degrees to radians
float quaternion::DegreesToRadians(float angle) {
	return angle * pi / 180.0f;
}

/*
v' = v/||v||
q_unit = [cos(o/2), sin(o/2)v']
*/
void quaternion::convertToRotationQuaternionRepresentation() {
	float rotationAngle = DegreesToRadians(s);
	s = std::cosf(rotationAngle / 2);
	vector.x *= std::sinf(rotationAngle / 2) / fastSquareRoot(dot(vector, vector));
	vector.y *= std::sinf(rotationAngle / 2) / fastSquareRoot(dot(vector, vector));
	vector.z *= std::sinf(rotationAngle / 2) / fastSquareRoot(dot(vector, vector));
}

/*
v' = v/||v||
transforms rotation representation from a unit quaternion :
q = cos(o/2)i_x + sin(o/2)v'_{y,z,w}
to a rotation matrix :
q = s + v-> R = |1 - 2 * (vy * vy + vz * vz)   2 * (vx * vy - s * vz)   2 * (vx * vz + s * vy)|
			    |2 * (vx * vy + s * vz)   1 - 2 * (vx * vx + vz * vz)   2 * (vy * vz - s * vx)|
			    |2 * (vx * vz - s * vy)   2 * (vy * vz + s * vx)   1 - 2 * (vx * vx + vy * vy)|
*/
std::vector<float> quaternion::getRotationMatrixFromUnitQuaternion(){
	// get vector
	float vx = vector.x; float vy = vector.y; float vz = vector.z;

	// return the matrix as a vector;
	return {1 - 2 * (vy * vy + vz * vz), 2 * (vx * vy - s * vz), 2 * (vx * vz + s * vy),
			2 * (vx * vy + s * vz), 1 - 2 * (vx * vx + vz * vz), 2 * (vy * vz - s * vx),
			2 * (vx * vz - s * vy), 2 * (vy * vz + s * vx), 1 - 2 * (vx * vx + vy * vy)};
}

/*
s = sqrt(1 + R0^2 + R4^2 + R8^2)/2
vx = (R7 - R5)/(4*s)
vy = (R2 - R6)/(4*s)
vz = (R3 - R1)/(4*s)
full explenation at : https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
*/
void quaternion::createUnitQuarenion(std::vector<float> rmat){
	// if the matrix trace > 0 then we can devide by s, else divition by
	// 0 will happend
	float trace = rmat[0] + rmat[4] + rmat[8];
	if(trace > 0){
		s = 0.5f * fastSquareRoot(1.0f + trace);
		vector.x = (rmat[7] - rmat[5]) / (4 * s);
		vector.y = (rmat[2] - rmat[6]) / (4 * s);
		vector.y = (rmat[3] - rmat[1]) / (4 * s);
	}
	else if(rmat[0] > rmat[4] && rmat[0] > rmat[8]){
		vector.x = 0.5f * fastSquareRoot(1.0f + rmat[0] - rmat[4] - rmat[8]);
		s = (rmat[7] - rmat[5]) / (4 * vector.x);
		vector.y = (rmat[3] + rmat[1]) / (4 * vector.x);
		vector.z = (rmat[2] + rmat[6]) / (4 * vector.x);
	}
	else if(rmat[4] > rmat[8]){
		vector.y = 0.5f * fastSquareRoot(1.0f + rmat[4] - rmat[0] - rmat[8]);
		s = (rmat[2] - rmat[6]) / (4 * vector.y);
		vector.x = (rmat[3] + rmat[1]) / (4 * vector.y);
		vector.z = (rmat[7] + rmat[5]) / (4 * vector.y);
	}
	else{
		vector.z = 0.5f * fastSquareRoot(1.0f + rmat[8] - rmat[0] - rmat[4]);
		s = (rmat[3] - rmat[1]) / (4 * vector.z);
		vector.x = (rmat[2] + rmat[6]) / (4 * vector.z);
		vector.y = (rmat[7] + rmat[5]) / (4 * vector.z);
	}
}