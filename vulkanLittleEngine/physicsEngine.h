#pragma once
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <array>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/pair.h>
#include <algorithm>

namespace MLPE {

	// rigid body physics
	namespace rbp {

		/* 
		single rigid body geomerty:
		vertices,
		indices,
		normals,
		particle decomosition,
		boundries
		*/

		/*
		quaternion representation of vectors in space, rotations, and general spacial transformations
		*/

		class MLPE_RBP_quaternion {
		public:

			// rotation + vector
			float s;
			glm::vec3 vector;

			// build quaternion
			MLPE_RBP_quaternion(float rotation, glm::vec3 D3Dvector) : s{ rotation }, vector{ D3Dvector } {}

			// overload - optional for defining a quaternion without inputs
			MLPE_RBP_quaternion() : s{ 0 }, vector{ glm::vec3(0) } {}

			// destructor
			~MLPE_RBP_quaternion(){}

			// basic operations
			/*
			operation +=
			*/
			void operator+=(const MLPE_RBP_quaternion& q);

			/*
			operation +
			*/
			MLPE_RBP_quaternion operator+(const MLPE_RBP_quaternion& q);

			/*
			operation -=
			*/
			void operator-=(const MLPE_RBP_quaternion& q);

			/*
			operation -
			*/
			MLPE_RBP_quaternion operator-(const MLPE_RBP_quaternion& q);

			/*
			operation *=
			*/
			void operator*=(const MLPE_RBP_quaternion& q);

			/*
			operation *
			*/
			MLPE_RBP_quaternion operator*(const MLPE_RBP_quaternion& q);

			/*
			scalar multiplication *=
			*/
			void operator*=(const float scale);

			/*
			scalar multiplication *
			*/
			MLPE_RBP_quaternion operator*(const float scale);

			// specified functions to use for 3d vector rotations

			/*
			L2 norm ||q||_2
			*/
			float Norm();

			/*
			q' = q/||q||
			*/
			void Normalize();

			/*
			q* = [s, -vector]
			*/
			void conjugate();
			MLPE_RBP_quaternion Conjugate();

			/*
			q^-1 = q* /||q||^2
			*/
			MLPE_RBP_quaternion inverse();

			// convert to rotation in 3d

			/*
			v' = v/||v||
			q_unit = [cos(o/2), sin(o/2)v']
			*/
			void ConvertToRotationQuaternionRepresentation();

		private:
			/*
			fast square root approximation
			*/
			float fastSquareRoot(float num);

			/*
			convert degrees in s to radians for trigonometric functions
			*/
			float DegreesToRadians(float angle);

		};

		/*
		general utility and info structs 
		an object is represented as a particle system where particles inside the object
		boundries do not interact with each other - the only interactions are between objects.
		massElements is the mass of a specific particle + the particle dimensions, s.t.
		
		massDistribution = std::vector<massElement> massElements // n particles
		float objectMass = thrust::reduce(massDistribution.m.begin(), massDistribution.m.end(), thrust::add<float>())
		float centerMass = massDistribution::getCenterMass(); (#)

		(#) centerMass = sum(r_i * m_i)/M

		geometry info containes the geometry of a given object - position in space
		*/

		struct particle {
			float radius;
			glm::vec3 center;
		};

		struct massElement {
			float m;
			particle particle;
		};


		struct mlpe_rbp_RigidBodyParticleDecomposition {
			std::vector<particle> particleDecomposition;
		};

		struct distribution {
			std::vector<float> prob;
		};

		struct mlpe_rbp_RigidBodyMassDistribution {
			std::vector<massElement> massElements;
		};

		struct mlpe_rbp_RigidBodyDynamicsInfo{
			// using thrust + glm
			/*
			---- constants ----
			-mass
			-geometry
			-Inertia tensor (of body)

			---- state of body (constantly computed) ----
			-position
			-time stamp
			-linear momentum
			-angular momentum
			-raw force queue
			-force decomposition : F = F_radial + F_tangent

			---- derived/computed quantities ----
			-linear velocity of mass center
			-center of mass
			-torque
			-angular acceleration
			-rotation axis + velocity(angular velocity)
			-rotation angle (degrees)
			*/
			MLPE_RBP_RigidBodyGeometryInfo geometricInfo;
			mlpe_rbp_RigidBodyParticleDecomposition particleDecomposition;

			float mass = 0;
			mlpe_rbp_RigidBodyMassDistribution massDistribution;
		};

		/*
		constructed operators for special thrust parallel operations
		*/

		// operator for summing over mass distribution structs - finding mass
		template<typename T>
		struct thrust_add_massElements : public thrust::binary_function<T, T, float> {
			__host__ __device__ float operator()(const T a, const T b)const {
				return a.m + b.m;
			}
		};

		// operator for summing over mass distribution structs - finding center of mass
		template<typename T>
		struct thrust_add_Positions : public thrust::binary_function<T, T, glm::vec3> {
			__host__ __device__ glm::vec3 operator()(const T& a, const T& b) {
				return a.particle.center * a.m + b.particle.center * b.m;
			}
		};

		// operator for comparing a value to one
		struct greater_than_one {
			__host__ __device__ bool operator()(float& a) {
				return a > 1;
			}
		};

		// operator for multiplying by a
		template<typename T>
		struct multiplyByConstant {
			// define the constnt (should be mass in massDistribution)
			const T constant;
			// constructor
			multiplyByConstant(T _c) : constant{ _c } {}

			__host__ __device__ float operator()(T vec_elem)const {
				return vec_elem * constant;
			}
		};

		/*
		essentially the same operator for comparing two values inside a vector of vectors
		*/

		// operator for finding extremum in certain direction
		template<typename T1, typename T2>
		struct extremumOp {
			// define a constant that affects inside operator
			const T2 type;
			// initialize struct - constructor
			extermumOp(T2 _t) : type{_t} {}

			__host__ __device__ bool operator()(T1 a, T1 b)const {
				return (type == "x") ? a.x < b.x : (type == "y") ? a.y < b.y : a.z < b.z;
			}
		};

		/*	
		-------------------- physics classes --------------------
		*/

		/*
		build a rigid body in space, and decompose into spherical particles
		*/
		class MLPE_RBP_RigidBodyGeometryInfo {
		public:
			std::vector<glm::vec3> vertices;
			std::vector<uint32_t> indices;

			// get all extrema points from all axis - minX, maxX, minY, maxY, minZ, maxZ
			std::array<thrust::pair<glm::vec3, uint32_t>, 6> getExtremumPoints();

		private:
			// diffranciate between minimum and maximum
			template<typename T>
			bool extremumType(T type);

			// copy from host to device
			template<typename T>
			thrust::device_vector<T> copy_vec(std::vector<T> vec);

			// find extremum along specific axis
			template<typename T1, typename T2>
			thrust::pair<glm::vec3, uint32_t> extremumAlongAxis(T1 typeOfExtremum, T2 axis);
		};

		class MLPE_RBP_RIGIDBODY_GEOMETRY {
		public:

			MLPE_RBP_RIGIDBODY_GEOMETRY(
				MLPE_RBP_RigidBodyGeometryInfo GeometricInfo,
				mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo) {

				loadGeometry(GeometricInfo, RigidBodyInfo);
				decomposeGeomerty(GeometricInfo);
				assignParticleDistribution(RigidBodyInfo);
			}
			~MLPE_RBP_RIGIDBODY_GEOMETRY() {};

		private:

			// loads geometry from 2
			void loadGeometry(
				MLPE_RBP_RigidBodyGeometryInfo GeometricInfo,
				mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo);

			// decompose model into particles
			void decomposeGeomerty(MLPE_RBP_RigidBodyGeometryInfo GeometricInfo);

			// assign coputed particle decomposition into Rigid body struct
			void assignParticleDistribution(mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo);

			/*
			produce a 3d grid that divides R3 into cubes of length 2*radius (of the particles)
			*/
			void get3Dgrid(MLPE_RBP_RigidBodyGeometryInfo GeometricInfo);

			mlpe_rbp_RigidBodyParticleDecomposition particleDecomposition;
		};

		/*
		physics classes - from more geometric to more physical:
		geometry->mass distribution->force distribution->velocity decomposition->inertia + acceleration->collisions + time stamping
		using particle system collistion detector and material representation
		using quaternion representation of points in space
		*/

		class MLPE_RBP_massDistribution {
		public:

			// needed to be initialized for the distribution function
			distribution massDistrib;

			MLPE_RBP_massDistribution(mlpe_rbp_RigidBodyDynamicsInfo RigidBodyInfo) {
				distributeMassElements(RigidBodyInfo);
				massElementsDistribution(RigidBodyInfo);
			}

			~MLPE_RBP_massDistribution(){}

			glm::vec3 getCenterMass(mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo);

		private:
			/*
			distribute mass elements according to object geometry - via particle mass distribution
			input needed to be: thrust::host/device<particle> particleDecomposition
			*/
			void distributeMassElements(mlpe_rbp_RigidBodyDynamicsInfo RigidBodyInfo);
			void mass(mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo);
			
			// copy vector from host vector to thrust device vector to be manipulated in parallel by GPU device 
			template<typename T>
			thrust::device_vector<T> copy_vec(std::vector<T> vec);

			// check for inappropriate distribution inputs
			template<typename T>
			void checkVector(std::vector<T> p);

			// assign computed mass distribution into Rigid body info struct
			void massElementsDistribution(mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo);

			mlpe_rbp_RigidBodyMassDistribution massDistribution;
		};


		/*
		.
		.
		.
		.
		.
		*/
		

		class MLPE_RBP_COLLISION_DETECTOR {
		public:
			/* MLPE_RBP_RIGIDBODY_GEOMETRY rigidBodyGeometry : to get rigidBodyGeometry.RigidBodyGeometricInfo()
			                                                          rigidBodyGeometry.RigidBodyParticleDecomposition() */ 
		};

		class MPE_RBP_RIGIDBODY {
		public:
		private:
			MLPE_RBP_RIGIDBODY_GEOMETRY geometryLoadOut;
			MLPE_RBP_massDistribution massDistribution;

			MLPE_RBP_RigidBodyGeometryInfo geometricInfo;
			mlpe_rbp_RigidBodyParticleDecomposition particleDecomposition;
		};

	}

	/*
	fluid dynamics
	*/
	namespace fd {

	}
}