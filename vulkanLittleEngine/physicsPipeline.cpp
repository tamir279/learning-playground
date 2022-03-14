#include <chrono>
#include <thread>
#include "physicsEngine.h"

namespace MLPE::pipeline {

	int tick = static_cast<int>(DT) * 1000;

	void MLPE_PIPELINE_PIPELINE::init() {
		for (auto body : bodies) {
			// set constructors
			// set geometry
			auto geometry = body.getGeometry();
			//geometry = rbp::MLPE_RBP_RIGIDBODY_GEOMETRY(body.getBodyInfo().geometricInfo, body.getBodyInfo())
			// set mass distribution
			auto massDistribution = body.getMassDistribution();
			massDistribution = rbp::MLPE_RBP_massDistribution(body.getBodyInfo(), body.M);
			body.setMassDistrib(massDistribution);
		}
	}

	void MLPE_PIPELINE_PIPELINE::mainLoop() {
		while (true) {
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