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
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <algorithm>
#include <stdexcept>
#include "physicsEngine.cuh"
#include "exceptionHandling.h"
/*
------------------- RIGID BODY PHYSICS -------------------
				 FORCES APPLIED AT TIME T
*/

namespace MLPE {
	namespace rbp {

		// for cleaner code
		//                   current body  outer body   contact point
		typedef thrust::tuple<massElement, massElement, glm::vec3> colPair;
		typedef mlpe_rbp_RigidBodyDynamicsInfo body;
		typedef std::vector<mlpe_rbp_RigidBodyDynamicsInfo> bodies;
		typedef std::vector<colPair> colVec;
		typedef thrust::device_vector<glm::vec3> devPtVec;
		typedef std::vector<glm::vec3> PtVec;
		typedef thrust::tuple<PtVec, PtVec, PtVec> PtVecTuple;

		/*
		calculate the collision impulse of each particle to particle interaction, neglecting interparticle
		forces, and particle angular velocity, i.e : for collision point i, bodies 1,2 : J_i = -(1+e)dot(v_r, n)/(1/m1 + 1/m2)
		for body pairs k = 1,2 :  Jk = J_i*(contactPt - p_k.center)
		notice : dot(v_r,n) = dot(v2 + w2xr2 - v1 - w1xr1, n). since ri = -+||ri||n, the dot product gives:
		dot(v_r,n) = dot(v2 - v1, n) + dot(w2xr2 - w1xr1, n) = dot(v2 - v1, n)
		*/

		// for "looping" over all collision PAIRS and collision points
		struct F : thrust::binary_function<colPair, glm::vec3, glm::vec3> {
			// constant of restitution
			const float e;
			// linear velocity of our body
			const glm::vec3 v1;
			// time delta
			const float dt;

			// constructor
			F(float _e, glm::vec3 _v, float _dt) : e{ _e }, v1{ _v }, dt{ _dt } {}

			// calculate J
			__host__ __device__
				glm::vec3 operator()(const colPair& collisionInfo, const glm::vec3& v2)const {
				// get contact point
				glm::vec3 cp = thrust::get<2>(collisionInfo);
				// particle masses
				float m1 = thrust::get<0>(collisionInfo).m;
				float m2 = thrust::get<1>(collisionInfo).m;
				// calculate impulse direction
				glm::vec3 J_hat = glm::normalize(thrust::get<0>(collisionInfo).particle.center - cp);
				// calculate the impulse magnitude
				float J_norm = (1 + e) * glm::abs(glm::dot(v1 - v2, J_hat)) / (1 / m1 + 1 / m2);
				// calculate force magnitude
				J_norm = J_norm / dt;

				return J_norm * J_hat;
			}
		};

		// get specific element of a tuple 
		template<typename T>
		struct getSpecific {
			// specific index
			const int ind;

			getSpecific(int _ind) : ind{ _ind } {}

			__host__ __device__
				T operator()(const thrust::tuple<T, T, T>& a) {
				return thrust::get<ind>(a);
			}
		};

		void getContactPtsAndOuterVelocities(colVec& contactPts, devPtVec& velocities, body bodyInfo, const bodies outerBodies) {
			for (auto body : outerBodies) {
				// the contact points
				colVec pts = bodyInfo.detector.detectCollisionObject_Object(bodyInfo, body);
				// create velocity container
				devPtVec velocityPerPt(pts.size());
				glm::vec3 v = (1 / body.mass) * thrust::get<2>(body.state.state);
				thrust::fill(thrust::device, velocityPerPt.begin(), velocityPerPt.end(), v);
				// concatenate results
				contactPts.insert(contactPts.end(), pts.begin(), pts.end());
				// take velocities
				velocities.insert(velocities.end(), velocityPerPt.begin(), velocityPerPt.end());
			}

			except::checkForEquality(velocities, contactPts);
		}

		void MLPE_RBP_ForceStateDiagram::checkForCollisionForces(
			mlpe_rbp_RigidBodyDynamicsInfo bodyInfo,
			const std::vector<mlpe_rbp_RigidBodyDynamicsInfo> outerBodies) {

			// check for each body for contact points
			colVec contactPts;
			devPtVec outerVelocities;
			// get contact point and outer center mass velocities 
			getContactPtsAndOuterVelocities(contactPts, outerVelocities, bodyInfo, outerBodies);

			// calculate forces
			// container
			PtVec forces;
			PtVec contacts;
			// body velocity before collision
			glm::vec3 v1 = (1 / bodyInfo.mass) * thrust::get<2>(bodyInfo.state.state);
			// calculate
			thrust::transform(
				thrust::device,
				thrust::device_pointer_cast(contactPts.data()),
				thrust::device_pointer_cast(contactPts.data()) + contactPts.size(),
				outerVelocities.begin(),
				thrust::device_pointer_cast(forces.data()),
				F(E, v1, DT));

			// get contacts
			thrust::transform(
				thrust::device,
				thrust::device_pointer_cast(contactPts.data()),
				thrust::device_pointer_cast(contactPts.data()) + contactPts.size(),
				thrust::device_pointer_cast(contacts.data()),
				getSpecific<glm::vec3>(2));
			// copy vectors
			CollisionForceDiagram = forces;
			contactPoints = contacts;
		}

		// TODO for later - when user inputs are defined
		void MLPE_RBP_ForceStateDiagram::checkForUserForceInput() {

		}

		void MLPE_RBP_ForceStateDiagram::checkIfGravityEnabled(mlpe_rbp_RigidBodyDynamicsInfo bodyInfo) {
			PtVec gravityDistribution;
			// check if gravity is enabled in the simulation
			glm::vec3 gravity = (bodyInfo.GravityEnabled) ? static_cast<float>(G) * glm::vec3(0, 0, -1.0f) : glm::vec3(0);

			thrust::fill_n(
				thrust::device,
				thrust::device_pointer_cast(gravityDistribution.data()),
				bodyInfo.particleDecomposition.particleDecomposition.size(),
				gravity);

			// transfer the data into the built in variable
			GravitationForceDiagram = gravityDistribution;
		}


		void MLPE_RBP_ForceStateDiagram::getForceState(
			mlpe_rbp_RigidBodyDynamicsInfo bodyInfo,
			const std::vector<mlpe_rbp_RigidBodyDynamicsInfo> outerBodies) {

			checkForCollisionForces(bodyInfo, outerBodies);
			checkForUserForceInput();
			checkIfGravityEnabled(bodyInfo);

			// padding sizes
			size_t padSize = bodyInfo.massDistribution.massElements.size();

			// neccesary padding
			GeneralUsage::padVector(InitialForceDiagram, glm::vec3(0), padSize);
			GeneralUsage::padVector(GravitationForceDiagram, glm::vec3(0), padSize);

			PtVecTuple forceDiag = thrust::make_tuple<PtVec, PtVec, PtVec>(
				CollisionForceDiagram,
				InitialForceDiagram,
				GravitationForceDiagram);

			ForceDistribution = forceDiag;
		}
	}
}