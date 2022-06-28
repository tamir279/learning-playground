#pragma once
#include <valarray>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// define pi
const float pi = 3.14159265359;

class quaternion {
public:

	// rotation + vector
	float s;
	float3 vector;

	// build quaternion
	quaternion(float rotation, float3 _vector){
		s = rotation;
		vector.x = _vector.x; 
		vector.y = _vector.y;
		vector.z = _vector.z;
	}

	// overload - optional for defining a quaternion without inputs
	quaternion() { s = 0; vector.x = 0; vector.y = 0; vector.z = 0; }

	// copy constructor
	quaternion(const quaternion& q){
		s = q.s;
		vector.x = q.vector.x; 
		vector.y = q.vector.y;
		vector.z = q.vector.z;
	}
	// assignment operator
	quaternion& operator=(const quaternion& q);
	// basic operations
	void operator+=(const quaternion& q);
	quaternion operator+(const quaternion& q);
	void operator-=(const quaternion& q);
	quaternion operator-(const quaternion& q);
	void operator*=(const quaternion& q);
	quaternion operator*(const quaternion& q);
	void operator*=(const float scale);
	quaternion operator*(const float scale);
	friend quaternion operator*(const float scale, const quaternion& q);
	// specified functions to use for 3d vector rotations

	/*
	L2 norm ||q||_2
	*/
	float Norm();
	void Normalize();

	/*
	q* = [s, -vector]
	*/
	void conjugate();
	quaternion Conjugate();

	/*
	q^-1 = q* /||q||^2
	*/
	quaternion inverse();

	// convert to rotation in 3d

	/*
	v' = v/||v||
	q_unit = [cos(o/2), sin(o/2)v']
	*/
	void convertToRotationQuaternionRepresentation();

	/*
	q = s + v-> R = |1 - 2 * (vy * vy + vz * vz)   2 * (vx * vy - s * vz)   2 * (vx * vz + s * vy)|
			        |2 * (vx * vy + s * vz)   1 - 2 * (vx * vx + vz * vz)   2 * (vy * vz - s * vx)|
			        |2 * (vx * vz - s * vy)   2 * (vy * vz + s * vx)   1 - 2 * (vx * vx + vy * vy)|
	*/
	std::vector<float> getRotationMatrixFromUnitQuaternion();

	/*
	R = |1 - 2 * (vy * vy + vz * vz)   2 * (vx * vy - s * vz)   2 * (vx * vz + s * vy)|
		|2 * (vx * vy + s * vz)   1 - 2 * (vx * vx + vz * vz)   2 * (vy * vz - s * vx)|
		|2 * (vx * vz - s * vy)   2 * (vy * vz + s * vx)   1 - 2 * (vx * vx + vy * vy)|

	=>  s = sqrt(1 + R{0,0}^2 + R{1,1}^2 + R{2,2}^2)/2
		vx = (R{2,1} - R{1,2})/(4*s)
		vy = (R{0,2} - R{2,0})/(4*s)
		vz = (R{1,0} - R{0,1})/(4*s)
	*/
	void createUnitQuarenion(std::vector<float> rmat);

private:
	float fastSquareRoot(float num);
	float DegreesToRadians(float angle);
};

