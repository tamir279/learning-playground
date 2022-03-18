#include <chrono>
#include <thread>
#include <vector>
#include "physicsEngine.cuh"

namespace MLPE::pipeline {

	int tick = static_cast<int>(DT) * 1000;

	void MLPE_PIPELINE_PIPELINE::init() {
		#pragma unroll(2)
		for (auto body : bodies) {
			// set constructors
			auto info = body.getBodyInfo();
			// set geometry
			auto geometry = body.getGeometry();
			geometry = rbp::MLPE_RBP_RIGIDBODY_GEOMETRY(info.geometricInfo, info);
			// set mass distribution
			auto massDistribution = body.getMassDistribution();
			massDistribution = rbp::MLPE_RBP_massDistribution(info, body.M);

			// get values after initialization
			body.setGeometry(geometry);
			body.setMassDistrib(massDistribution);
		}
	}

	void MLPE_PIPELINE_PIPELINE::mainLoop() {
		while (true) {
			#pragma unroll(2)
			for (auto body : bodies) {
				auto tmpBodyArr = GeneralUsage::eraseElement(bodies, body);
				auto info = body.getBodyInfo();
				body.CurrentState.step(info, tmpBodyArr);
			}
			/*
			pipeline:
			init : create all initial bodies
			mainLoop : steps, create/delete bodies
			*/
			// rigidBody.state.step()
			std::this_thread::sleep_for(std::chrono::milliseconds(tick));
		}
	}
}