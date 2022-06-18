#include <stdexcept>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/remove.h>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>

/*
device allocations are relevant only with device vectors (memory access using device_vector iterators)
*/

/*
---------------------------- reduce operator ----------------------------
*/

template <typename T, typename inputOperator, typename BinaryFunction>
__host__ __device__ T thrust_wrapper_reduce(
											bool COMPUTE_IN_DEVICE,
											inputOperator first,
											inputOperator last,
											T init,
											BinaryFunction binaryOperator
											) {
	// for readability
	typedef thrust::device_vector<T>::iterator device_iterator;

	if (typeid(first) != typeid(last)) throw std::runtime_error("iterator types do not match");

	if (COMPUTE_IN_DEVICE && typeid(first) == typeid(device_iterator) ) {
		return thrust::reduce(thrust::device, first, first, init, binaryOperator);
	}
	else if (COMPUTE_IN_DEVICE) {
		thurst::device_vector<T> dv(first, last);
		return thrust::reduce(thrust::device, dv.begin(), dv.end(), init, binaryOperator);
	}	
	return thrust::reduce(thrust::host, first, last, init, binaryOperator);
}

/*
---------------------------- transform operator ---------------------------- 
*/

// overload 1 
template <typename inputOperator, typename outputOperator, typename UnaryFunction>
__host__ __device__ outputOperator thrust_wrapper_transform(
															bool COMPUTE_IN_DEVICE,
														    inputOperator first,
														    inputOperator last,
										                    outputOperator result,
												            UnaryFunction unaryOperator
											                ) {
	// for readability
	typedef thrust::device_vector<inputOperator::value_type>::iterator device_iterator;

	if (typeid(first) != typeid(last)) throw std::runtime_error("iterator types do not match");

	if (COMPUTE_IN_DEVICE && typeid(first) == typeid(device_iterator)) {
		return thrust::transform(thrust::device, first, last, result, unaryOperator);
	}
	else if (COMPUTE_IN_DEVICE) {
		thurst::device_vector<inputOperator::value_type> dv(first, last);
		return thrust::transform(thrust::device, dv.begin(), dv.end(), result, unaryOperator);
	}
	return thrust::transform(thrust::host, first, last, result, unaryOperator);
}

// overload 2
template <typename inputOperator1, typename inputOperator2, typename outputOperator, typename BinaryFunction>
__host__ __device__ outputOperator thrust_wrapper_transform(
															bool COMPUTE_IN_DEVICE,
															inputOperator1 first1,
															inputOperator1 last1,
															inputOperator2 first2,
														    inputOperator2 last2,
															outputOperator result,
															BinaryFunction BinaryOperator
															) {
	// for readability
	typedef thrust::device_vector<inputOperator1::value_type>::iterator device_iterator1;
	typedef thrust::device_vector<inputOperator2::value_type>::iterator device_iterator2;
	
	// device vectors for possible needed device allocation
	thurst::device_vector<inputOperator1::value_type> dv1(first1, last1);
	thurst::device_vector<inputOperator2::value_type> dv2(first2, last2);
	
	if (typeid(first1) != typeid(last1) || typeid(first2) != typeid(last2)) 
		throw std::runtime_error("iterator types do not match");

	if (COMPUTE_IN_DEVICE && typeid(first1) == typeid(device_iterator1) && typeid(first2) == typeid(device_iterator2)) {
		return thrust::transform(thrust::device, first1, last1, first2, result, BinaryOperator);
	}
	else if (COMPUTE_IN_DEVICE && typeid(first1) != typeid(device_iterator1) && typeid(first2) == typeid(device_iterator2)) {
		return thrust::transform(thrust::device, dv1.begin(), dv1.end(), first2, result, BinaryOperator);
	}
	else if (COMPUTE_IN_DEVICE && typeid(first1) == typeid(device_iterator1) && typeid(first2) != typeid(device_iterator2)) {
		return thrust::transform(thrust::device, first1, last1, dv2.begin(), result, BinaryOperator);
	}
	else if (COMPUTE_IN_DEVICE) {
		return thrust::transform(thrust::device, dv1.begin(), dv1.end(), dv2.begin(), result, BinaryOperator);
	}
	return thrust::transform(thrust::host, first1, last1, first2, result, BinaryOperator);
}

/*
---------------------------- extremum operators ----------------------------
*/

// max element
// overload 1
template<typename ForwardIterator>
__host__ __device__ ForwardIterator thrust_wrapper_max_element(
																bool COMPUTE_IN_DEVICE,
																ForwardIterator first,
																ForwardIterator last
																) {
	typedef thurst::device_vector<ForwardIterator::value_type>::iterator device_it;
	
	if (typeid(first) != typeid(last)) throw std::runtime_error("iterator types do not match");
	
	if (COMPUTE_IN_DEVICE && typeid(first) == typeid(device_it)) {
		return thrust::max_element(thrust::device, first, last);
	}
	else if (COMPUTE_IN_DEVICE) {
		thurst::device_vector<ForwardIterator::value_type> dv(first, last);
		return thrust::max_element(thrust::device, dv.begin(), dv.end());
	}
	return thrust::max_element(thrust::host, first, last);
}

// max element
// overload 2
template<typename ForwardIterator, typename BinaryFunction>
__host__ __device__ ForwardIterator thrust_wrapper_max_element(
																bool COMPUTE_IN_DEVICE,
																ForwardIterator first,
																ForwardIterator last,
																BinaryFunction BinaryOperator
																) {
	typedef thurst::device_vector<ForwardIterator::value_type>::iterator device_it;

	if (typeid(first) != typeid(last)) throw std::runtime_error("iterator types do not match");

	if (COMPUTE_IN_DEVICE && typeid(first) == typeid(device_it)) {
		return thrust::max_element(thrust::device, first, last, BinaryOperator);
	}
	else if (COMPUTE_IN_DEVICE) {
		thurst::device_vector<ForwardIterator::value_type> dv(first, last);
		return thrust::max_element(thrust::device, dv.begin(), dv.end(), BinaryOperator);
	}
	return thrust::max_element(thrust::host, first, last, BinaryOperator);
}

// min element
// overload 1
template<typename ForwardIterator>
__host__ __device__ ForwardIterator thrust_wrapper_min_element(
																bool COMPUTE_IN_DEVICE,
																ForwardIterator first,
																ForwardIterator last
																) {
	typedef thurst::device_vector<ForwardIterator::value_type>::iterator device_it;

	if (typeid(first) != typeid(last)) throw std::runtime_error("iterator types do not match");

	if (COMPUTE_IN_DEVICE && typeid(first) == typeid(device_it)) {
		return thrust::min_element(thrust::device, first, last);
	}
	else if (COMPUTE_IN_DEVICE) {
		thurst::device_vector<ForwardIterator::value_type> dv(first, last);
		return thrust::max_element(thrust::device, dv.begin(), dv.end());
	}
	return thrust::min_element(thrust::host, first, last);
}

// min element
// overload 2
template<typename ForwardIterator, typename BinaryFunction>
__host__ __device__ ForwardIterator thrust_wrapper_min_element(
																bool DEVICE,
																ForwardIterator first,
																ForwardIterator last,
																BinaryFunction BinaryOperator
																) {
	typedef thurst::device_vector<ForwardIterator::value_type>::iterator device_it;

	if (typeid(first) != typeid(last)) throw std::runtime_error("iterator types do not match");

	if (COMPUTE_IN_DEVICE && typeid(first) == typeid(device_it)) {
		return thrust::min_element(thrust::device, first, last, BinaryOperator);
	}
	else if (COMPUTE_IN_DEVICE) {
		thurst::device_vector<ForwardIterator::value_type> dv(first, last);
		return thrust::min_element(thrust::device, dv.begin(), dv.end(), BinaryOperator);
	}
	return thrust::min_element(thrust::host, first, last, BinaryOperator);
}

/*
---------------------------- remove operators ----------------------------
*/

// remove copy if
// overload 1
template<typename inputIterator, typename outputIterator, typename Predicate>
__host__ __device__ outputIterator thrust_wrapper_remove_copy_if(
																 bool DEVICE,
																 inputIterator first,
																 inputIterator last,
																 outputIterator result,
																 Predicate pred
																 ) {
	if (DEVICE) {
		thurst::device_vector<inputIterator::value_type> dv(first, last);
		return thrust::remove_copy_if(thrust::device, dv.begin(), dv.end(), result, pred);
	}
	return thrust::remove_copy_if(thrust::host, first, last, result, pred);
}

// remove copy if
// overload 2
template<typename inputIterator1, typename inputIterator2, typename outputIterator, typename Predicate>
__host__ __device__ outputIterator thrust_wrapper_remove_copy_if(
															     bool DEVICE,
																 inputIterator1 first,
																 inputIterator1 last,
																 inputIterator2 stencilFirst,
																 inputIterator2 stencilLast,
																 outputIterator result,
																 Predicate pred
																 ) {
	if (DEVICE) {
		thurst::device_vector<inputIterator1::value_type> dv(first, last);
		thurst::device_vector<inputIterator2::value_type> dvStencil(stencilFirst, stencilLast);
		return thrust::remove_copy_if(thrust::device, dv.begin(), dv.end(), dvStencil.begin(), result, pred);
	}
	return thrust::remove_copy_if(thrust::host, first, last, stencilFirst, result, pred);
}

/*
---------------------------- fill operator ----------------------------
*/

template<typename ForwardIterator, typename T>
__host__ __device__ void thrust_wrapper_fill(
											 bool DEVICE,
											 ForwardIterator first,
											 ForwardIterator last,
											 const T& value
											 ) {
	if (DEVICE) {
		thurst::device_vector<T> dv(first, last);
		thrust::fill(thrust::device, dv.begin(), dv.end(), value);
		thrust::copy(dv.begin(), dv.end(), first);
	}
	else {
		thrust::fill(thrust::host, first, last, value);
	}
}

/*
---------------------------- fill_n operator ----------------------------
*/

template<typename outputIterator, typename size, typename T>
__host__ __device__ outputIterator thrust_wrapper_fill_n(
														 bool DEVICE,
														 outputIterator first,
														 outputIterator last,
														 size n,
														 const T& value
														 ) {
	if (DEVICE) {
		thurst::device_vector<T> dv(first, last);
		thrust::fill_n(thrust::device, dv.begin(), n, value);
		thrust::copy(dv.begin(), dv.end(), first);
	}
	else {
		thrust::fill_n(thrust::host, first, n, value);
	}
}

/*
---------------------------- sequence operator ----------------------------
*/

// overload 1
template<typename ForwardIterator>
__host__ __device__ void thrust_wrapper_sequence(
												 bool DEVICE,
												 ForwardIterator first,
												 ForwardIterator last
												 ) {
	if (DEVICE) {
		thurst::device_vector<ForwardIterator::value_type> dv(first, last);
		thrust::sequence(thrust::device, dv.begin(), dv.end());
		thrust::copy(dv.begin(), dv.end(), first);
	}
	else {
		thrust::sequence(thrust::host, first, last);
	}
}

// overload 2
template<typename ForwardIterator, typename T>
__host__ __device__ void thrust_wrapper_sequence(
												 bool DEVICE,
												 ForwardIterator first,
												 ForwardIterator last,
												 T init
												 ) {
	if (DEVICE) {
		thurst::device_vector<T> dv(first, last);
		thrust::sequence(thrust::device, dv.begin(), dv.end(), init);
		thrust::copy(dv.begin(), dv.end(), first);
	}
	else {
		thrust::sequence(thrust::host, first, last, init);
	}
}

// overload 3
template<typename ForwardIterator, typename T>
__host__ __device__ void thrust_wrapper_sequence(
												 bool DEVICE,
												 ForwardIterator first,
												 ForwardIterator last,
												 T init,
												 T step
												 ) {
	if (DEVICE) {
		thurst::device_vector<T> dv(first, last);
		thrust::sequence(thrust::device, dv.begin(), dv.end(), init, step);
		thrust::copy(dv.begin(), dv.end(), first);
	}
	else {
		thrust::sequence(thrust::host, first, last, init, step);
	}
}

/*
---------------------------- tabulate operator ----------------------------
*/

template<typename ForwardIterator, typename UnaryOperation>
__host__ __device__ void thrust_wrapper_tabulate(
												 bool DEVICE,
												 ForwardIterator first,
												 ForwardIterator last,
												 UnaryOperation unaryOperator
												 ) {
	if (DEVICE) {
		thrust::device_vector<ForwardIterator::value_type> dv(first, last);
		thrust::tabulate(thrust::device, dv.begin(), dv.end(), unaryOperator);
		thrust::copy(dv.begin(), dv.end(), first);
	}
	else thrust::tabulate(thrust::host, first, last, unaryOperator);
}