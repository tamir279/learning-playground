#pragma once
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>
#include <cmath>

class quaternion {
public:

	// rotation + vector
	float s;
	glm::vec3 vector;

	// build quaternion
	quaternion(float rotation, glm::vec3 D3Dvector) : s{ rotation }, vector{ D3Dvector } {}

	// overload - optional for defining a quaternion without inputs
	quaternion() : s{ 0 }, vector{ glm::vec3(0) } {}

	// destructor
	~quaternion() {}

	// basic operations
	void operator+=(const quaternion& q);
	quaternion operator+(const quaternion& q);
	void operator-=(const quaternion& q);
	quaternion operator-(const quaternion& q);
	void operator*=(const quaternion& q);
	quaternion operator*(const quaternion& q);
	void operator*=(const float scale);
	quaternion operator*(const float scale);

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
	void ConvertToRotationQuaternionRepresentation();

private:
	float fastSquareRoot(float num);
	float DegreesToRadians(float angle);
};