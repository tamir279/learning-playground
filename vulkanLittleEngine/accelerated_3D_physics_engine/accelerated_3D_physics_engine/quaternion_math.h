#pragma once
#include <valarray>
#include <stdexcept>
#include <vector>
#include <cmath>

// define pi
const float pi = 3.14159265359;

class quaternion {
public:

	// rotation + vector
	float s;
	std::valarray<float> vector = std::valarray<float>(3);

	// build quaternion
	quaternion(float rotation, std::valarray<float> _vector){
		s = rotation;
		vector = _vector;
	}

	// overload - optional for defining a quaternion without inputs
	quaternion() { s = 0; }

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

