#include <stdexcept>
#include <vector>

#include "exceptionHandling.h"

namespace except {
	// exception checking
	template<typename... Args>
	void checkForEquality(const Args&... args) {
		if constexpr (sizeof...(args) > 1) {
			auto vec = { args... };
			// unroll two iterations - only sensible when the number of arguments is bigger then 1
			#pragma unroll(2)
			for (auto it = vec.begin(); it != vec.end(); ++it) {
				if ((*it).size() != (*(vec.begin())).size())
					throw std::runtime_error("argument sizes are not equal! cannot create thrust::iterators!");
			}
		}
	}

	// check for specific size of an array
	template<typename T1, typename T2>
	void checkSizeOfArray(const T1 expectedSize, const T2& arr) {
		auto arrSize = arr.size();
		if (arrSize > 0) {
			if(static_cast<T1>(arrSize) != expectedSize)
				throw std::runtime_error("array size is not correct");
		}
		else {
			throw std::runtime_error("array is empty!");
		}
	}

	template<typename T>
	void checkIfEnded(T it, T end) {
		if (it != end) throw::std::runtime_error("scanning stopped before end of vector");
	}
}