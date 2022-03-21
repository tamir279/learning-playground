#pragma once
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>
#include <vector>
#include <array>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/pair.h>
#include <algorithm>

#define PARTICE_RADIUS 0.1
#define DT 1E-02
#define G 9.81
#define E 0.5

namespace MLPE {

	namespace GeneralUsage {

		// general use - transform from vector to thrust::device_vector
		template<typename T>
		thrust::device_vector<T> mlpe_gu_copyVector(std::vector<T> vec) {
			thrust::host_vector<T> th_vec;
			thrust::copy(vec.begin(), vec.end(), th_vec.begin());
			thrust::device_vector<T> res = th_vec;

			return res;
		}

		// general use - transform thrust::vector to vector
		template<typename T>
		std::vector<T> mlpe_gu_copyBackVector(thrust::device_vector<T> vec) {
			std::vector<T> th_vec;
			thrust::copy(vec.begin(), vec.end(), th_vec.begin());
			std::vector<T> res = th_vec;

			return res;
		}

		// general use - 3D determinant - for floats/integers only
		float mlpe_gu_3Ddeterminant(glm::vec3 a, glm::vec3 b, glm::vec3 c) {
			glm::mat3 colMat = glm::mat3(a, b, c);
			return glm::determinant(colMat);
		}

		// an OR binary operator
		template<typename T>
		struct OR : public thrust::binary_function <T, T, T> {
			__host__ __device__ T operator()(T a, T b) {
				return a | b;
			}
		};

		// an AND binary operator
		template<typename T>
		struct AND : public thrust::binary_function<T, T, T> {
			__host__ __device__ T operator()(T a, T b) {
				return a & b;
			}
		};

		// a summation operator
		template<typename T>
		struct Plus : public thrust::binary_function<T, T, T> {
			__host__ __device__ T operator()(T a, T b) {
				return a + b;
			}
		};

		// rotate a vector along specific axis by using a unit quaternion
		struct mlpe_gu_rotate : public thrust::unary_function<glm::vec3, glm::vec3> {
			const rbp::MLPE_RBP_quaternion rotQuaternion;

			mlpe_gu_rotate(rbp::MLPE_RBP_quaternion _q) : rotQuaternion{ _q } {}

			__host__ __device__ glm::vec3 operator()(glm::vec3 particleCenter) const {
				// defining the particle quaternion for rotation
				rbp::MLPE_RBP_quaternion pQuaternion(180, particleCenter);
				// gives pQuaternion = [cos(180/2), sin(180/2)*particleCenter] = [0, particleCenter]
				pQuaternion.ConvertToRotationQuaternionRepresentation();
				// calculate
				rbp::MLPE_RBP_quaternion rQuaternion = rotQuaternion;
				pQuaternion = rQuaternion * pQuaternion * rQuaternion.inverse();
				return pQuaternion.vector;
			}
		};

		// unary addition - the type T have be defined with a '+' operation
		template<typename T>
		struct mlpe_gu_Uadd : public thrust::unary_function<T, T> {
			const T a1;
			mlpe_gu_Uadd(T _a1) : a1{ _a1 } {}

			__host__ __device__ T operator()(T a2) {
				return a1 + a2;
			}
		};

		// rotate and translate particles in one operator
		struct mlpe_gu_rotateAndTranslate : public thrust::unary_function<glm::vec3, rbp::particle> {
			const rbp::MLPE_RBP_quaternion rotQuaternion;
			const glm::vec3 center;

			mlpe_gu_rotateAndTranslate(rbp::MLPE_RBP_quaternion _q, glm::vec3 _tr) : rotQuaternion{ _q }, center{ _tr } {}

			__host__ __device__ rbp::particle operator()(const glm::vec3& particleCenter) const {
				// defining the particle quaternion for rotation
				rbp::MLPE_RBP_quaternion pQuaternion(180, particleCenter);
				// gives pQuaternion = [cos(180/2), sin(180/2)*particleCenter] = [0, particleCenter]
				pQuaternion.ConvertToRotationQuaternionRepresentation();
				// calculate
				rbp::MLPE_RBP_quaternion rQuaternion = rotQuaternion;
				pQuaternion = rQuaternion * pQuaternion * rQuaternion.inverse();

				// get the particle wih the new center location
				rbp::particle p{ PARTICE_RADIUS , pQuaternion.vector + center };
				return p;
			}
		};

		// pad a vector with specific value - pad until v has a specific length
		template<typename T>
		auto padVector(std::vector<T>& v, T pad, size_t a) {
			std::vector<T> padVec(a);
			thrust::fill(
				thrust::device,
				thrust::device_pointer_cast(padVec.data),
				thrust::device_pointer_cast(padVec.data) + a,
				pad - v.size());
			v.insert(v.end(), padVec.begin(), padVec.end());
		}

		// erase specific value and return the new vector
		template<typename T>
		auto eraseElement(std::vector<T> arr, T elem) {
			std::vector<T> tmp = arr;
			tmp.erase(std::remove(tmp.begin(), tmp.end(), elem), tmp.end());
			return tmp;
		}
		/*
		// create struct from arbitrary number of types , and structs
		template<typename... T>
		struct gather {

		};
		*/
	}


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
			~MLPE_RBP_quaternion() {}

			// basic operations
			void operator+=(const MLPE_RBP_quaternion& q);
			MLPE_RBP_quaternion operator+(const MLPE_RBP_quaternion& q);
			void operator-=(const MLPE_RBP_quaternion& q);
			MLPE_RBP_quaternion operator-(const MLPE_RBP_quaternion& q);
			void operator*=(const MLPE_RBP_quaternion& q);
			MLPE_RBP_quaternion operator*(const MLPE_RBP_quaternion& q);
			void operator*=(const float scale);
			MLPE_RBP_quaternion operator*(const float scale);

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
			float fastSquareRoot(float num);
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

		// support is limited to triangle polygons
		struct polygon {
			thrust::tuple<glm::vec3, glm::vec3, glm::vec3> polygon;
		};

		struct particle {
			float radius;
			glm::vec3 center;
		};

		struct massElement {
			float m;
			particle particle;
		};

		// for grid creation with thrust::tabulate;

		// 3D grid 
		struct Object3DCubeGrid {
			//             x dir      y dir     z dir
			thrust::tuple<uint32_t, uint32_t, uint32_t> gridAxisSize;
			uint32_t gridSize;
			std::vector<glm::vec3> grid;
		};

		struct mlpe_rbp_RigidBodyParticleDecomposition {
			std::vector<particle> particleDecomposition;
		};

		struct distribution {
			std::vector<float> prob;
		};

		// TODO : assert particleDecomposition.size() == massElements.size()
		struct mlpe_rbp_RigidBodyMassDistribution {
			std::vector<massElement> massElements;
			glm::vec3 centerMass;
		};

		struct mlpe_rbp_RigidBodyDynamicsInfo {
			// time
			// simulation time - maximum number of time steps is 2^64 - 1 ~ 18*10E19
			uint64_t t_n = 0;

			// constants
			MLPE_RBP_RigidBodyGeometryInfo geometricInfo;
			mlpe_rbp_RigidBodyParticleDecomposition particleDecomposition;

			float mass = 0;
			mlpe_rbp_RigidBodyMassDistribution massDistribution;

			// state of body
			// initial inertia tensor
			glm::mat3 I0 = glm::mat3(0);
			bodyState state;
			bool GravityEnabled = true;

			// collision detection
			MLPE_RBP_COLLISION_DETECTOR detector;
			// force diagram
			MLPE_RBP_ForceStateDiagram forceState;
		};

		struct bodyState {
			//          cm position /  rotation /  linear momentum / angular momentum
			thrust::tuple<glm::vec3, MLPE_RBP_quaternion, glm::vec3, glm::vec3> state;
			// inverse inertial tensor / force sum / torque / angular velocity
			thrust::tuple<glm::mat3, glm::vec3, glm::vec3, glm::vec3> auxilaryState;
			// forces state
			thrust::tuple<
				std::vector<glm::vec3>,
				std::vector<glm::vec3>,
				std::vector<glm::vec3>> forceDiagram;
			// contact points
			std::vector<glm::vec3> contactPts;
			// initial body system positions of particle centers
			std::vector<glm::vec3> r0;
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
			extremumOp(T2 _t) : type{ _t } {}

			__host__ __device__ bool operator()(T1 a, T1 b)const {
				return (type == "x") ? a.x < b.x : (type == "y") ? a.y < b.y : a.z < b.z;
			}
		};

		// operator for massDistribution combination of data
		struct mElementComb {
			__host__ __device__ massElement operator()(float d, particle p) {
				massElement mE;
				mE.m = d; mE.particle = p;
				return mE;
			}
		};

		// operator struct for calculating signed solid angle of a polygon compared to specific point
		struct solidAngle : public thrust::unary_function<polygon, float> {
			// the reference point
			const glm::vec3 p;
			// p is an input
			solidAngle(glm::vec3 _p) : p{ _p } {}

			__host__ __device__ float operator()(polygon P)const {
				/*
				tan(solidAngle/2) = det(a b c)/(|a||b||c| + (a*b)|c| + (b*c)|a| + (c*a)|b|)
				=> solidAngle = 2arctan(det(a b c)/(|a||b||c| + (a*b)|c| + (b*c)|a| + (c*a)|b|))
				=> factoredSolidAngle = (1/2PI)*arctan(det(a b c)/(|a||b||c| + (a*b)|c| + (b*c)|a| + (c*a)|b|))
				*/

				// definitions of a, b, c
				glm::vec3 a = thrust::get<0>(P.polygon) - p;
				glm::vec3 b = thrust::get<1>(P.polygon) - p;
				glm::vec3 c = thrust::get<2>(P.polygon) - p;

				// calculating norms
				float n_a = glm::l2Norm(a);
				float n_b = glm::l2Norm(b);
				float n_c = glm::l2Norm(c);

				// calculating determinant
				float det_abc = GeneralUsage::mlpe_gu_3Ddeterminant(a, b, c);
				float monster = n_a * n_b * n_c + n_c * glm::dot(a, b) + n_a * glm::dot(b, c) + n_b * glm::dot(a, c);

				return glm::atan(det_abc, monster);
			}
		};

		// operator for taking only the second parameter of a thrust pair
		template<typename T1, typename T2>
		struct secondArgument {
			__host__ __device__ T2 operator()(thrust::pair<T1, T2> a) {
				return thrust::get<1>(a);
			}
		};

		// operator for substracting vectors - calculating position in body coordinates
		struct minus {
			// with relation with center of mass
			const glm::vec3 centerMass;
			// constructor
			minus(glm::vec3 _cm) : centerMass{ _cm } {}

			__host__ __device__ glm::vec3 operator()(particle p) {
				return p.center - centerMass;
			}
		};

		// operator to calculate kernal matrix of inertia tensor
		struct kernel : public thrust::binary_function<glm::vec3, massElement, glm::mat3> {
			__host__ __device__ glm::mat3 operator()(glm::vec3 r, massElement m)const {
				// m((r^T*r)I - r*r^T)
				return m.m * (glm::length2(r) * glm::mat3(1.0f) - glm::outerProduct(r, r));
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

			// polygon structure from file
			std::vector<polygon> objPolygons;

			// define the particle size (radius)
			float r = static_cast<float>(PARTICE_RADIUS);

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

			// get grid size
			void get3DgridSize(thrust::tuple<uint32_t, uint32_t, uint32_t> gridSizes);

			/*
			in order ot decompose geometry we need to decide which particles are inside of the object
			*/
			bool calcSignedngleForSpecificPoint(glm::vec3 p, std::vector<polygon> polygons);

			void isParticleInsideObject(MLPE_RBP_RigidBodyGeometryInfo GeometricInfo, std::vector<particle>& DC);

			// define 3d grid and particle vector for the object
			Object3DCubeGrid grid;
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

			MLPE_RBP_massDistribution(mlpe_rbp_RigidBodyDynamicsInfo RigidBodyInfo, float mass) {
				Mass(RigidBodyInfo, mass);
				distributeMassElements(RigidBodyInfo);
				getCenterMass(RigidBodyInfo);
				massElementsDistribution(RigidBodyInfo);
			}

			~MLPE_RBP_massDistribution() {}

		private:
			/*
			distribute mass elements according to object geometry - via particle mass distribution
			input needed to be: thrust::host/device<particle> particleDecomposition
			*/
			void distributeMassElements(mlpe_rbp_RigidBodyDynamicsInfo RigidBodyInfo);
			void Mass(mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo, float mass);
			void getCenterMass(mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo);

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

		// detect collisions and return contact points (if any exist)
		class MLPE_RBP_COLLISION_DETECTOR {
		public:

			std::vector<thrust::tuple<massElement, massElement, glm::vec3>> detectCollisionObject_Object(
				mlpe_rbp_RigidBodyDynamicsInfo OuterObjectInfo,
				mlpe_rbp_RigidBodyDynamicsInfo ObjectInfo);

		private:

			// plot collision points between one particle of object and another object
			std::vector<thrust::tuple<bool, massElement, massElement, glm::vec3>> P_O_checkCollisionPoints(
				massElement m,
				mlpe_rbp_RigidBodyDynamicsInfo& OuterObjectInfo);
		};

		// queue for forces at different times, diagram at time t of all the forces applied on the body
		class MLPE_RBP_ForceStateDiagram {
		public:

			// force applied on each particle - to calculate change in rotation
			thrust::tuple<
				std::vector<glm::vec3>,
				std::vector<glm::vec3>,
				std::vector<glm::vec3>> ForceDistribution;
			// all contact points
			std::vector<glm::vec3> contactPoints;

			// for raisig new request for searching outer forces applied on body
			void getForceState(
				mlpe_rbp_RigidBodyDynamicsInfo bodyInfo,
				const std::vector<mlpe_rbp_RigidBodyDynamicsInfo> outerBodies);

		private:
			void checkForCollisionForces(
				mlpe_rbp_RigidBodyDynamicsInfo bodyInfo,
				const std::vector<mlpe_rbp_RigidBodyDynamicsInfo> outerBodies);
			void checkForUserForceInput();
			void checkIfGravityEnabled(mlpe_rbp_RigidBodyDynamicsInfo bodyInfo);

			// for collision points - up to maxFroceCapacity = number of particles
			std::vector<glm::vec3> CollisionForceDiagram;
			// for initial forces
			std::vector<glm::vec3> InitialForceDiagram;
			// for gravity
			std::vector<glm::vec3> GravitationForceDiagram;
		};

		// calculates the physical state of a rigid body at time t (step t/DT)
		class MLPE_RBP_rigidBodyState {
		public:
			// time delta
			float dt = (float)DT;

			bodyState state_n;

			// IMPORTANT
			// state_n.r0 = state_n_m_1.r0;

			MLPE_RBP_rigidBodyState() {}
			~MLPE_RBP_rigidBodyState() {}

			void step(
				mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo,
				const std::vector<mlpe_rbp_RigidBodyDynamicsInfo> outerBodies) {
				if (!RigidBodyInfo.t_n) {
					initializeState(RigidBodyInfo);
				}
				else {

					if (RigidBodyInfo.t_n == UINT64_MAX) { RigidBodyInfo.t_n = 0; }

					getPreviousState(RigidBodyInfo);
					calculateForceDistribution(RigidBodyInfo, outerBodies);
					calculateCenterMass();
					calculateRotationQuaternion();
					calculateParticleCenter(RigidBodyInfo);
					calculateLinearMomentum();
					calculateAngularMomentum();
					calculateTotalForce();
					calculateTorque(RigidBodyInfo);
					calculateInverseInertiaTensor(RigidBodyInfo);
					calculateAngularVelocity();
				}
				updateState(RigidBodyInfo);
			}

		private:
			// initalize state of body - at time 0
			void initializeState(mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo);

			// get state at time t - dt
			void getPreviousState(const mlpe_rbp_RigidBodyDynamicsInfo RigidBodyInfo);

			// state at time t
			void calculateCenterMass();
			void calculateParticleCenter(mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo);
			void calculateRotationQuaternion();
			void calculateLinearMomentum();
			void calculateAngularMomentum();
			void calculateForceDistribution(
				mlpe_rbp_RigidBodyDynamicsInfo bodyInfo,
				const std::vector<mlpe_rbp_RigidBodyDynamicsInfo> outerBodies);
			void calculateTotalForce();
			void calculateTorque(mlpe_rbp_RigidBodyDynamicsInfo RigidBodyInfo);
			void calculateInverseInertiaTensor(mlpe_rbp_RigidBodyDynamicsInfo RigidBodyInfo);
			void calculateAngularVelocity();

			// update state
			void updateState(mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo);

			// the body state at n-1
			bodyState state_n_m_1;
			// body mass - an input when 
			float M;
		};

		/*
		.
		.
		.
		.
		.
		*/



		class MLPE_RBP_RIGIDBODY {
		public:
			MLPE_RBP_rigidBodyState CurrentState;

			float M;

			void setBodyInfo(mlpe_rbp_RigidBodyDynamicsInfo _Info) { Info = _Info; }
			auto getBodyInfo() { return Info; }

			void setGeometry(MLPE_RBP_RIGIDBODY_GEOMETRY _gL) { geometryLoader = _gL; }
			auto getGeometry() { return geometryLoader; }

			void setMassDistrib(MLPE_RBP_massDistribution _mD) { massDistribution = _mD; }
			auto getMassDistribution() { return massDistribution; }
		private:
			mlpe_rbp_RigidBodyDynamicsInfo Info;
			MLPE_RBP_RIGIDBODY_GEOMETRY geometryLoader;
			MLPE_RBP_massDistribution massDistribution;
		};

	}

	/*
	fluid dynamics
	*/
	namespace fd {
		// class MLPE_FD_FLUID_PARTICLE
	}

	// for the main pipeline
	namespace pipeline {

		class MLPE_PIPELINE_PIPELINE {
		public:
			void init();
			void mainLoop();
		private:
			std::vector<rbp::MLPE_RBP_RIGIDBODY> bodies;
		};
	}
}