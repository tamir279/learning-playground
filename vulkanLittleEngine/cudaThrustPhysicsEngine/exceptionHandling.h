#pragma once
#include <stdexcept>
#include <vector>

namespace except {
	// equality checking of multiple values of certain types
	template<typename... Args>
	void checkForEquality(const Args&... args);

	// check for equality to specific object
	template<typename T1, typename T2>
	void checkSizeOfArray(const T1 expectedSize, const T2& arr);

	template<typename T>
	void checkIfEnded(T it, T end);

}