#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <array>

#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include<thrust/transform_scan.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <algorithm>
#include <stdexcept>
#include "physicsEngine.cuh"

/*
------------------- RIGID BODY PHYSICS -------------------
		RIGID BODY PHYSICAL STATE AT SPECIFIC TIME T
*/

namespace MLE::MLPE {
	namespace rbp {

		// for cleaner code
		typedef thrust::device_vector<glm::vec3> DevicePtVector;
		typedef thrust::device_vector<particle> DeviceParticleVec;
		typedef std::vector<particle> ParticleVec;
		typedef thrust::tuple<DevicePtVector, DevicePtVector, DevicePtVector> devPtVecTuple;
		typedef thrust::tuple<particle, glm::vec3> particleTuple;
		typedef thrust::tuple<glm::vec3, massElement> geometryTuple;
		typedef std::vector<glm::vec3> PtVector;

		// summation for thrust zip iterator - 3 individual iterators
		template<typename T>
		struct zipPlus {
			__host__ __device__ T operator()(const thrust::tuple<T, T, T>& a)const {
				return thrust::get<0>(a) + thrust::get<1>(a) + thrust::get<2>(a);
			}
		};

		// summation for thrust zip iterator - 3 individual iterators
		template<typename T>
		struct zipBinaryPlus : thrust::binary_function<thrust::tuple<T, T>, thrust::tuple<T, T>, T> {
			__host__ __device__ T operator()(const thrust::tuple<T, T, T>& a, const thrust::tuple<T, T>& b)const {

				T cross_a = glm::cross(thrust::get<0>(a), thrust::get<1>(a));
				T cross_b = glm::cross(thrust::get<0>(b), thrust::get<1>(b));

				return cross_a + cross_b;
			}
		};

		// summation of particle centers cross forces
		struct zipBinaryParticles {
			__host__ __device__ glm::vec3 operator()(const particleTuple& a, const particleTuple& b) {

				glm::vec3 cross_a = glm::cross(thrust::get<0>(a).center, thrust::get<1>(a));
				glm::vec3 cross_b = glm::cross(thrust::get<0>(b).center, thrust::get<1>(b));

				return cross_a + cross_b;
			}
		};

		// operator to calculate kernal matrix of inertia tensor - Unary form
		struct kernelUnary : public thrust::unary_function<geometryTuple, glm::mat3> {
			__host__ __device__ glm::mat3 operator()(const geometryTuple tuple)const {
				// get components
				glm::vec3 r = thrust::get<0>(tuple);
				massElement m = thrust::get<1>(tuple);
				// m((r^T*r)I - r*r^T)
				return m.m * (glm::length2(r) * glm::mat3(1.0f) - glm::outerProduct(r, r));
			}
		};

		// calculate Icm 
		glm::mat3 calculateInitialInertiaTensor(
			glm::vec3 cm,
			std::vector<particle> pd,
			std::vector<massElement> me) {

			DevicePtVector relativeDistance;
			// get relative distance to center of mass
			thrust::transform(
				thrust::device,
				thrust::device_pointer_cast(pd.data()),
				thrust::device_pointer_cast(pd.data()) + pd.size(),
				relativeDistance.begin(),
				minus(cm));

			/*
			calculate kernel matrix for Inertia tensor - m((r^T*r)I - r*r^T) and sum over it,
			i.e. I = sum(m((r^T*r)I - r*r^T)) = sum(kernelMatrix)
			*/
			return thrust::reduce(
				thrust::device,
				thrust::make_transform_iterator(thrust::make_zip_iterator(relativeDistance.begin(), me.begin()), kernelUnary()),
				thrust::make_transform_iterator(thrust::make_zip_iterator(relativeDistance.end(), me.end()), kernelUnary()),
				glm::mat3(0),
				GeneralUsage::Plus<glm::mat3>());
		}

		/*
		calculate rotation matrix from unit quaternion q = [cos(t/2), sin(t/2)*n] = [q1, q2, q3, q4]

		q1 = cos(t/2)
		q2 = sin(t/2)n_x
		q3 = sin(t/2)n_y
		q4 = sin(t/2)n_z

		R =   [[1-2q2^2 - 2q3^2 , 2q1q2 - 2q0q3 , 2q1q3 + 2q0q2],
			   [2q1q2 + 2q0q3 , 1-2q1^2 - 2q3^2 , 2q2q3 - 2q0q1],
			   [2q1q3 - 2q0q2 , 2q2q3 + 2q0q1 , 1-2q1^2 - 2q2^2]]
		*/

		glm::mat3 convertQuaternionToRotationMatrix(MLPE_RBP_quaternion q) {
			// convert the quaternion into a unit quaternion
			q.ConvertToRotationQuaternionRepresentation();
			// map q components
			float q0 = q.s;
			float q1 = q.vector.x;
			float q2 = q.vector.y;
			float q3 = q.vector.z;
			// populate the rotation matrix
			return glm::mat3(1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2),
							 2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q0 * q1),
							 2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2));
		}

		// t = 0
		void MLPE_RBP_rigidBodyState::initializeState(mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo) {
			// get particle decomposition
			auto decomp = RigidBodyInfo.particleDecomposition.particleDecomposition;
			// main parameters of a body state
			// r_cm
			thrust::get<0>(state_n.state) = RigidBodyInfo.massDistribution.centerMass;
			// r0
			thrust::transform(
				thrust::device,
				thrust::device_pointer_cast(decomp.data()),
				thrust::device_pointer_cast(decomp.data()) + decomp.size(),
				thrust::device_pointer_cast(state_n.r0.data()),
				minus(RigidBodyInfo.massDistribution.centerMass));
			// q_obj
			MLPE_RBP_quaternion q(0, glm::vec3(0, 0, 1));
			q.ConvertToRotationQuaternionRepresentation();
			thrust::get<1>(state_n.state) = q;
			// P_cm
			thrust::get<2>(state_n.state) = glm::vec3(0, 0, 0);
			// L_cm
			thrust::get<3>(state_n.state) = glm::vec3(0, 0, 0);

			// auxilary parameters
			// initial inertia tensor
			RigidBodyInfo.I0 = glm::inverse(calculateInitialInertiaTensor(
				RigidBodyInfo.massDistribution.centerMass,
				decomp,
				RigidBodyInfo.massDistribution.massElements));
			// inverse inertia tensor
			thrust::get<0>(state_n.auxilaryState) = RigidBodyInfo.I0;
			/*
			force state
			*/
			PtVector gForce;
			PtVector collForce;
			PtVector InitForce;
			// get gravity
			glm::vec3 gravity = (RigidBodyInfo.GravityEnabled) ? static_cast<float>(G) * glm::vec3(0, 0, -1.0f) : glm::vec3(0);
			// fill
			thrust::fill_n(
				thrust::device,
				thrust::device_pointer_cast(gForce.data()),
				decomp.size(),
				gravity);
			collForce.push_back(glm::vec3(0));
			InitForce.push_back(glm::vec3(0));

			thrust::get<0>(state_n.forceDiagram) = collForce;
			thrust::get<1>(state_n.forceDiagram) = InitForce;
			thrust::get<2>(state_n.forceDiagram) = gForce;
			/*
			total force
			*/
			thrust::get<1>(state_n.auxilaryState) = thrust::reduce(
				thrust::device,
				thrust::device_pointer_cast(gForce.data()),
				thrust::device_pointer_cast(gForce.data()) + gForce.size(),
				glm::vec3(0),
				GeneralUsage::Plus<glm::vec3>());
			/*
			torque
			*/
			// angular velocity
			thrust::get<3>(state_n.auxilaryState) = glm::vec3(0, 0, 1);
		}

		void MLPE_RBP_rigidBodyState::getPreviousState(const mlpe_rbp_RigidBodyDynamicsInfo RigidBodyInfo) {
			state_n_m_1 = RigidBodyInfo.state;
			M = RigidBodyInfo.mass;
		}

		void MLPE_RBP_rigidBodyState::calculateForceDistribution(
			mlpe_rbp_RigidBodyDynamicsInfo bodyInfo,
			const std::vector<mlpe_rbp_RigidBodyDynamicsInfo> outerBodies) {

			bodyInfo.forceState.getForceState(bodyInfo, outerBodies);
			state_n.forceDiagram = bodyInfo.forceState.ForceDistribution;
			state_n.contactPts = bodyInfo.forceState.contactPoints;
		}

		// translate body
		void MLPE_RBP_rigidBodyState::calculateCenterMass() {
			// r_n = r_(n-1) + DT*v_(n-1) = r_(n-1) + (DT/M)*P_(n-1)
			thrust::get<0>(state_n.state) = thrust::get<0>(state_n_m_1.state) + (dt / M) * thrust::get<2>(state_n_m_1.state);
		}

		// TODO : check if it is needed to notmalize and transform into rotation quaternion
		void MLPE_RBP_rigidBodyState::calculateRotationQuaternion() {
			// angular velocity quaternion from previous state
			MLPE_RBP_quaternion w(0, thrust::get<3>(state_n_m_1.auxilaryState));
			// previous rotation velocity
			MLPE_RBP_quaternion q_n = thrust::get<1>(state_n_m_1.state);
			// q_n+1 = q_n + DT/2 w_n*q_n
			thrust::get<1>(state_n.state) = q_n + (w * q_n) * (dt / 2.0f);
		}

		// r_new = r_cm + q_r*r0*q_r^-1
		// rotate body
		void MLPE_RBP_rigidBodyState::calculateParticleCenter(mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo) {
			ParticleVec updatedLocations(state_n.r0.size());
			// updated center mass position (body translation)
			glm::vec3 Rcm = thrust::get<0>(state_n.state);
			// rotating the initial relative centers + adding centerMass
			thrust::transform(
				thrust::device,
				thrust::device_pointer_cast(state_n.r0.data()),
				thrust::device_pointer_cast(state_n.r0.data()) + state_n.r0.size(),
				updatedLocations.begin(),
				GeneralUsage::mlpe_gu_rotateAndTranslate(thrust::get<1>(state_n.state), Rcm));

			// copy vector to particleDecomposition
			RigidBodyInfo.particleDecomposition.particleDecomposition = updatedLocations;
		}


		void MLPE_RBP_rigidBodyState::calculateLinearMomentum() {
			// P_n+1 = P_n + DT*F_n
			thrust::get<2>(state_n.state) = thrust::get<2>(state_n_m_1.state) + dt * thrust::get<1>(state_n_m_1.auxilaryState);
		}

		void MLPE_RBP_rigidBodyState::calculateAngularMomentum() {
			// L_n+1 = L_n + DT*Tau_n
			thrust::get<3>(state_n.state) = thrust::get<3>(state_n_m_1.state) + dt * thrust::get<2>(state_n_m_1.auxilaryState);
		}

		void MLPE_RBP_rigidBodyState::calculateTotalForce() {
			// get data and device pointers
			auto Fc = thrust::get<0>(state_n.forceDiagram); thrust::device_ptr<glm::vec3> dFc(Fc.data());
			auto Fi = thrust::get<1>(state_n.forceDiagram); thrust::device_ptr<glm::vec3> dFi(Fi.data());
			auto Fg = thrust::get<2>(state_n.forceDiagram); thrust::device_ptr<glm::vec3> dFg(Fg.data());

			// get last place device pointers
			thrust::device_ptr<glm::vec3> dFcE = dFc + Fc.size();
			thrust::device_ptr<glm::vec3> dFiE = dFi + Fi.size();
			thrust::device_ptr<glm::vec3> dFgE = dFg + Fg.size();

			// pad to identical size - for later use of iterators
			GeneralUsage::padVector<glm::vec3>(Fc, glm::vec3(0), Fi.size());

			// computing results individually, and sum over all forces
			thrust::get<1>(state_n.auxilaryState) = 
				thrust::reduce(
					thrust::device,
					thrust::make_transform_iterator(thrust::make_zip_iterator(dFc, dFi, dFg), zipPlus<glm::vec3>()),
					thrust::make_transform_iterator(thrust::make_zip_iterator(dFc, dFiE, dFgE), zipPlus<glm::vec3>()),
					glm::vec3(0),
					GeneralUsage::Plus<glm::vec3>());
		}

		// TODO : improve the solution by uniting the two calculations into one big zip iterated computation
		void MLPE_RBP_rigidBodyState::calculateTorque(mlpe_rbp_RigidBodyDynamicsInfo RigidBodyInfo) {
			auto Fc = thrust::get<0>(state_n.forceDiagram); thrust::device_ptr<glm::vec3> dFc(Fc.data());
			auto Fi = thrust::get<1>(state_n.forceDiagram); thrust::device_ptr<glm::vec3> dFi(Fi.data());
			auto Fg = thrust::get<2>(state_n.forceDiagram); thrust::device_ptr<glm::vec3> dFg(Fg.data());

			// computing torque elements for collision points
			// calculate first result
			glm::vec3 contactRes = 
				thrust::reduce(
					thrust::device,
					thrust::make_zip_iterator(thrust::device_pointer_cast(state_n.contactPts.data()), dFc),
					thrust::make_zip_iterator(thrust::device_pointer_cast(state_n.contactPts.data()) + state_n.contactPts.size(), dFc + Fc.size()),
					glm::vec3(0),
					zipBinaryPlus<glm::vec3>());

			// compute torque elements for gravity
			glm::vec3 gravityRes = glm::vec3(0);

			// calculate the result only if there i substential data inside
			if (RigidBodyInfo.GravityEnabled) {
				auto particleFIter = thrust::device_pointer_cast(RigidBodyInfo.particleDecomposition.particleDecomposition.data());
				auto particleLIter = particleFIter + RigidBodyInfo.particleDecomposition.particleDecomposition.size();

				gravityRes = 
					thrust::reduce(
						thrust::device,
						thrust::make_zip_iterator(particleFIter, dFg),
						thrust::make_zip_iterator(particleLIter, dFg + Fg.size()),
						glm::vec3(0),
						zipBinaryParticles());
			}
			thrust::get<2>(state_n.auxilaryState) = gravityRes + contactRes;
		}

		void MLPE_RBP_rigidBodyState::calculateInverseInertiaTensor(mlpe_rbp_RigidBodyDynamicsInfo RigidBodyInfo) {
			MLPE_RBP_quaternion q_n = thrust::get<1>(state_n.state);
			glm::mat3 R = convertQuaternionToRotationMatrix(q_n);
			thrust::get<0>(state_n.auxilaryState) = R * RigidBodyInfo.I0 * glm::transpose(R);
		}

		void MLPE_RBP_rigidBodyState::calculateAngularVelocity() {
			// w_n+1 = I_n+1 * L_n+1
			thrust::get<3>(state_n.auxilaryState) = thrust::get<0>(state_n.auxilaryState) * thrust::get<3>(state_n.state);
		}

		void MLPE_RBP_rigidBodyState::updateState(mlpe_rbp_RigidBodyDynamicsInfo& RigidBodyInfo) {
			// update state
			RigidBodyInfo.state = state_n;
			// update time step
			RigidBodyInfo.t_n += 1;
		}
	}
}