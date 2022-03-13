#include <chrono>
#include <thread>
#include "physicsEngine.h"

namespace MLPE::pipeline {
	void mainLoop() {
		for (uint64_t i = 0; i < UINT64_MAX; i++) {
			/*
			pipeline:
			init : create all initial bodies
			mainLoop : steps, create/delete bodies
			*/
			// rigidBody.state.step()
			std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(DT * 1000.0f)));
		}
	}
}