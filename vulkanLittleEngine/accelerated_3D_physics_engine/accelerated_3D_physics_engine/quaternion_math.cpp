#include "quaternion_math.h"

// basic vector operations
// dot product
float dot(std::valarray<float> v1, std::valarray<float> v2){
	return (v1 * v2).sum();
}

// cross product
std::valarray<float> cross(std::valarray<float> v1, std::valarray<float> v2){
	return { v1[1] * v2[2] - v2[1] * v1[2], v2[0] * v1[2] - v1[1] * v2[2], v1[0] * v2[1] - v2[0] * v1[1] };
}

// library functions

void quaternion::operator+=(const quaternion& q) {
	s += q.s;
	vector += q.vector;
}

quaternion quaternion::operator+(const quaternion& q) {
	quaternion q_res;
	q_res.s = s + q.s;
	q_res.vector = vector + q.vector;

	return q_res;
}

void quaternion::operator-=(const quaternion& q) {
	s -= q.s;
	vector -= q.vector;
}

quaternion  quaternion::operator-(const quaternion& q) {
	quaternion q_res;
	q_res.s = s - q.s;
	q_res.vector = vector - q.vector;

	return q_res;
}

/*
quat *= q <=> quat = quat * q

quat = [s, vector], q = [s', vector']
=> quat = [s*s' - dot(vector, vector'), s*vector' + s'*vector + cross(vector, vector')
*/
void quaternion::operator*=(const quaternion& q) {
	s = s * q.s - dot(vector, q.vector);
	vector = q.vector * s + vector * q.s + cross(vector, q.vector);
}

/*
res = quat * q
*/
quaternion quaternion::operator*(const quaternion& q) {
	quaternion q_res;
	q_res.s = s * q.s - dot(vector, q.vector);
	q_res.vector = q.vector * s + vector * q.s + cross(vector, q.vector);

	return q_res;
}

// scalar multiplication
void quaternion::operator*=(const float scale) {
	s *= scale;
	vector *= scale;
}

quaternion quaternion::operator*(const float scale) {
	quaternion q_res;
	q_res.s = s * scale;
	q_res.vector = vector * scale;

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
	s /= Norm();
	vector *= 1.0f / Norm();
}

/*
q* = [s, -v]
*/
void quaternion::conjugate() {
	vector *= -1.0f;
}

quaternion quaternion::Conjugate() {
	quaternion q_conjugate;
	q_conjugate.s = s;
	q_conjugate.vector = vector * (-1.0f);

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
	vector *= std::sinf(rotationAngle / 2) / fastSquareRoot(dot(vector, vector));
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
	float vx = vector[0]; float vy = vector[1]; float vz = vector[2];

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
		vector[0] = (rmat[7] - rmat[5]) / (4 * s);
		vector[1] = (rmat[2] - rmat[6]) / (4 * s);
		vector[2] = (rmat[3] - rmat[1]) / (4 * s);
	}
	else if(rmat[0] > rmat[4] && rmat[0] > rmat[8]){
		vector[0] = 0.5f * fastSquareRoot(1.0f + rmat[0] - rmat[4] - rmat[8]);
		s = (rmat[7] - rmat[5]) / (4 * vector[0]);
		vector[1] = (rmat[3] + rmat[1]) / (4 * vector[0]);
		vector[2] = (rmat[2] + rmat[6]) / (4 * vector[0]);
	}
	else if(rmat[4] > rmat[8]){
		vector[1] = 0.5f * fastSquareRoot(1.0f + rmat[4] - rmat[0] - rmat[8]);
		s = (rmat[2] - rmat[6]) / (4 * vector[1]);
		vector[0] = (rmat[3] + rmat[1]) / (4 * vector[1]);
		vector[2] = (rmat[7] + rmat[5]) / (4 * vector[1]);
	}
	else{
		vector[2] = 0.5f * fastSquareRoot(1.0f + rmat[8] - rmat[0] - rmat[4]);
		s = (rmat[3] - rmat[1]) / (4 * vector[2]);
		vector[0] = (rmat[2] + rmat[6]) / (4 * vector[2]);
		vector[1] = (rmat[7] + rmat[5]) / (4 * vector[2]);
	}
}