#include <cstring>
#include <cstdio>
#include <typeinfo>
#include <stdexcept>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cusparseLt.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cuComplex.h>

__global__ void printMeDev(float* P, int m, int n) {;
	for (int i = 0; i < m; i++) {
		printf("\n");
		for (int j = 0; j < n; j++)
			printf("%f ", P[j*m + i]);
	}
	printf("\n");
}

__global__ void printArr_di32(int32_t* P, int l) {
	for (int i = 0; i < l; i++) {
		printf("%d ", (int)P[i]);
	}
	printf("\n");
}

__global__ void printArr_df32(float* P, int l) {
	for (int i = 0; i < l; i++) {
		printf("%f ", P[i]);
	}
	printf("\n");
}

/*
---------------------------------------------------------------------------------------------------
----------------------------------------CUDA ERROR DETECTION---------------------------------------
---------------------------------------------------------------------------------------------------
*/

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

#define checkCuBLAS_status(ans) { cublasAssert((ans), __FILE__, __LINE__); }

#define checkCuSparseErrors(call)                                     \
  do {                                                                \
    cusparseStatus_t status = call;                                   \
    if (status != CUSPARSE_STATUS_SUCCESS) {                          \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n",  \
               __LINE__, cusparseGetErrorString(status), status);     \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

#define checkCuSolverErrors(call)                                     \
  do {                                                                \
    cusolverStatus_t status = call;                                   \
    if (status != CUSOLVER_STATUS_SUCCESS) {                          \
      printf("CUSOLVER API failed at line %d with error: %d\n",       \
               __LINE__,  status);                                    \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

#define HOST_ALLOC(memLoc)((memLoc) == memLocation::HOST_PINNED || (memLoc) == memLocation::HOST)
#define DEVICE_ALLOC(memLoc)((memLoc) == memLocation::DEVICE)

#define TILE_DIM 8
#define BLOCK_ROWS 8
// general error definition

inline void cublasAssert(cublasStatus_t code, const char* file, int line, bool abort = true)
{
	if (code != CUBLAS_STATUS_SUCCESS)
	{
		switch (code) {
		case CUBLAS_STATUS_NOT_INITIALIZED:
			fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_NOT_INITIALIZED file: %s line: %d ", file, line);
			break;

		case CUBLAS_STATUS_ALLOC_FAILED:
			fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_ALLOC_FAILED file: %s line: %d ", file, line);
			break;

		case CUBLAS_STATUS_INVALID_VALUE:
			fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_INVALID_VALUE file: %s line: %d ", file, line);
			break;

		case CUBLAS_STATUS_ARCH_MISMATCH:
			fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_ARCH_MISMATCH file: %s line: %d ", file, line);
			break;

		case CUBLAS_STATUS_MAPPING_ERROR:
			fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_MAPPING_ERROR file: %s line: %d ", file, line);
			break;

		case CUBLAS_STATUS_EXECUTION_FAILED:
			fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_EXECUTION_FAILED file: %s line: %d ", file, line);
			break;

		case CUBLAS_STATUS_INTERNAL_ERROR:
			fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_INTERNAL_ERROR file: %s line: %d ", file, line);
			break;

		case CUBLAS_STATUS_NOT_SUPPORTED:
			fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_NOT_SUPPORTED file: %s line: %d ", file, line);
			break;

		case CUBLAS_STATUS_LICENSE_ERROR:
			fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_LICENSE_ERROR file: %s line: %d ", file, line);
			break;
		}
		if (abort) exit(code);
	}
}

// generic function for handling GPU memory releases 
template<typename... args>
void cudaFreeMem(args*... rawPts) {
	auto arg_vec = { rawPts... };
	for (auto elem : arg_vec) {
		checkCudaErrors(cudaFree(elem));
	}
}

// generic function for copying memory with an asynchronous fasion between gpu and cpu memory with
// cudaMemcpyAsync - copying to specific type
template<typename dataType>
void asyncMemcopy(std::vector<dataType*> i_list,
				  std::vector<dataType*> o_list, 
				  std::vector<size_t> sizes,
				  std::vector<cudaMemcpyKind> kinds,
				  int numCopies){
	// initialize streams
	std::vector<cudaStream_t> streamz(numCopies);
	for(auto& stream : streamz){
		checkCudaErrors(cudaStreamCreate(&stream));
	}

	// copy - check if lengths are identical
	if(i_list.size() != numCopies || o_list.size() != numCopies || kinds.size() != numCopies){
		throw std::length_error("iput sizes do not match expected copies to be made");
	}
	for(int i = 0; i < numCopies; i++){
		checkCudaErrors(cudaMemcpyAsync(o_list[i], i_list[i], sizes[i], kinds[i], streamz[i]));
	}

	// synchronize device - block device from runnig more instructions before host code
	checkCudaErrors(cudaDeviceSynchronize());

	// destroy streamz
	for(auto& stream : streamz){
		checkCudaErrors(cudaStreamDestroy(stream));
	}
}

/*
---------------------------------------------------------------------------------------------------
-------------------------------------------MATRIX CLASS--------------------------------------------
---------------------------------------------------------------------------------------------------
*/

// three memory maps
enum class memLocation { HOST, HOST_PINNED, DEVICE };
// factorization type
enum DECOMP {LU, QR, CHOL};

// matrix class in column major order
template<typename T>
class mat {
public:
	// thee matrix data
	T* data;
	// dimensions
	int M; int N;
	memLocation memState;
	// empty or not
	bool empty = false;

	mat(int n_rows, int n_cols, memLocation memLoc) {
		// inputs
		this->M = n_rows;
		this->N = n_cols;
		this->memState = memLoc;

		// allocate memory
		if (HOST_ALLOC(memState)) {
			allocateOnHost(&data, M, N, memState);
		}
		else {
			allocateOnDevice(&data, M, N);
		}
	}

	// empty constructor
	mat() {
		empty = true;
	}

	// copy constructor
	mat(const mat<T>& obj) {
		// copy parameters
		this->M = obj.M;
		this->N = obj.N;
		this->memState = obj.memState;

		// allocate memory
		(HOST_ALLOC(memState)) ? allocateOnHost(&data, M, N, memState) : allocateOnDevice(&data, M, N);

		// copy data
		dynamicCopy(data, obj.data, M, N, memState, obj.memState);
	}

	~mat() {
		if (!empty) {
			(HOST_ALLOC(memState)) ? destroyHostMatrix(&data, memState) : destroyDeviceMatrix(&data);
		}
	}

	// operators
	mat<T>& operator+=(const mat<T>& B) {
		if (empty) throw std::length_error("empty data ");
		// allocate result matrix on device for cublas computation
		mat<T> C(M, N, memLocation::DEVICE);

		matAdd(CUBLAS_OP_N,
			   CUBLAS_OP_N,
			   M, N,
			   (const T**)&data, M,
			   memState,
			   (const T**)&(B.data), M,
		       B.memState,
			   &(C.data), M,
			   C.memState);

		// copy back to this matrix
		(HOST_ALLOC(memState))
			? copyDeviceToHost(C.data, data, M, N) : copyDeviceToDevice(C.data, data, M, N);
		return *this;
	}

	// the + operator returns a device allocated matrix
	mat<T> operator+(const mat<T>& B) {
		// allocate result matrix on device for cublas computation
		mat<T> C(M, N, memLocation::DEVICE);

		matAdd(CUBLAS_OP_N,
			   CUBLAS_OP_N,
			   M, N,
			   (const T**)&data, M,
			   memState,
			   (const T**)&(B.data), M,
			   B.memState,
			   &(C.data), M,
			   C.memState);

		// return the new matrix
		return C;
	}

	mat<T>& operator*=(const mat<T>& B) {
		if (empty) throw std::length_error("empty data ");
		// allocate result matrix on device for cublas computation
		mat<T> C(M, B.N, memLocation::DEVICE);

		//get the cuda types
		cudaDataType mT = m_T();

		// C(M,N) = A(M,N)B(N,B.N)
		matMul(CUBLAS_OP_N,
			   CUBLAS_OP_N,
			   M, B.N, N,
			   (const T**)&data, mT,
			   memState,
			   M,
			   (const T**)&(B.data), mT,
			   B.memState,
			   N,
			   &(C.data), mT,
			   M, mT,
			   CUBLAS_GEMM_DEFAULT);

		// copy back to this matrix
		(HOST_ALLOC(memState) && N == C.N)
			? copyDeviceToHost(C.data, data, M, N) : copyDeviceToDevice(C.data, data, M, N);
		return *this;
	}

	// the * operator returns a device allocated matrix
	mat<T> operator*(const mat<T>& B) {
		// allocate result matrix on device for cublas computation
		mat<T> C(M, B.N, memLocation::DEVICE);

		//get the cuda types
		cudaDataType mT = m_T();

		// C(M,N) = A(M,N)B(N,B.N)'
		matMul(CUBLAS_OP_N,
			   CUBLAS_OP_N,
			   M, B.N, N,
			   (const T**)&data, mT,
			   memState,
			   M,
			   (const T**)&(B.data), mT,
			   B.memState,
			   N,
			   &(C.data), mT,
			   M, mT,
			   CUBLAS_GEMM_DEFAULT);

		checkCudaErrors(cudaDeviceSynchronize());
		// return the new matrix
		return C;
	}

	// copy host to device - from B to matrix m
	mat<T>& operator<<=(const mat<T>& B) {
		emptyAllocation(&data, B.M, B.N, B.memState, empty, &M, &N);
		copyHostToDevice(B.data, data, M, N);
		return *this;
	}

	// copy device to host - from B to matrix m
	mat<T>& operator>>=(const mat<T>& B) {
		emptyAllocation(&data, B.M, B.N, B.memState, empty, &M, &N);
		copyDeviceToHost(B.data, data, M, N);
		return *this;
	}

	// copy host to host - from B to matrix m
	mat<T>& operator<=(const mat<T>& B) {
		emptyAllocation(&data, B.M, B.N, B.memState, empty, &M, &N);
		copyHostToHost(B.data, data, M, N);
		return *this;
	}

	// copy device to device - from B to matrix m
	mat<T>& operator>=(const mat<T>& B) {
		emptyAllocation(&data, B.M, B.N, B.memState, empty, &M, &N);
		copyDeviceToDevice(B.data, data, M, N);
		return *this;
	}

	// dynamic copying
	mat<T>& operator=(const mat<T>& B) {
		emptyAllocation(&data, B.M, B.N, B.memState, empty, &M, &N);
		dynamicCopy(data, B.data, M, N, memState, B.memState);
		return *this;
	}

	// check if transfer to device is needed
	bool checkToAllocateOnDevice(const T* h_m, T** d_m, int n_rows, int n_cols, memLocation memLoc);

	// decide compute and data types for matrix m - m_T matrix data type, c_T - compute type
	cudaDataType m_T();

	// special matrix functions
	// transpose - using tiled transpose kernel
	void transpose();

private:
	// memory and device management ( + overloads)
	// ver. 1 - no const

	// allocation
	void allocateOnHost(T** h_m, int n_rows, int n_cols, memLocation memLoc);
	void allocateOnDevice(T** d_m, int n_rows, int n_cols);
	void emptyAllocation(T** m, int n_rows, int n_cols, memLocation memLoc, bool empty_mat, int* r, int* c);

	// copy
	void copyHostToDevice(T* h_m, T* d_m, int n_rows, int n_cols);
	void copyHostToHost(T* h_m1, T* h_m2, int n_rows, int n_cols);
	void copyDeviceToHost(T* d_m, T* h_m, int n_rows, int n_cols);
	void copyDeviceToDevice(T* d_m1, T* d_m2, int n_rows, int n_cols);

	// ver. 2 - const src - copy
	void copyHostToDevice(const T* h_m, T* d_m, int n_rows, int n_cols);
	void copyHostToHost(const T* h_m1, T* h_m2, int n_rows, int n_cols);
	void copyDeviceToHost(const T* d_m, T* h_m, int n_rows, int n_cols);
	void copyDeviceToDevice(const T* d_m1, T* d_m2, int n_rows, int n_cols);

	// dynamic copy
	void dynamicCopy(T* data1, T* data2, int n_rows, int n_cols, memLocation memLoc1, memLocation memLoc2);
	// overload
	//void dynamicCopy(mat<T> mat1, mat<T> mat2, int n_rows, int n_cols);

	// matrix addition
	void matAdd(cublasOperation_t transa,
				cublasOperation_t transb,
				int m, int n,
				const T** A, int lda,
				memLocation Aloc,
				const T** B, int ldb,
				memLocation Bloc,
				T** C, int ldc,
				memLocation Cloc);

	// matrix multiplication
	void matMul(cublasOperation_t transa,
				cublasOperation_t transb,
				int m, int n, int k,
				const T** A,
				cudaDataType Atype,
				memLocation Aloc,
				int lda,
				const T** B,
				cudaDataType Btype,
				memLocation Bloc,
				int ldb,
				T** C,
				cudaDataType Ctype,
				int ldc,
				cudaDataType computeType,
				cublasGemmAlgo_t algo);

	// memory cleanup
	void destroyHostMatrix(T** h_m, memLocation memLoc);
	void destroyDeviceMatrix(T** d_m);
};

// return the cuda data type according to type T of m
template<typename T>
cudaDataType mat<T>::m_T() {
	// type to be returned
	cudaDataType type;

	(typeid(T) == typeid(float)) ? type = CUDA_R_32F :
	(typeid(T) == typeid(double)) ? type = CUDA_R_64F :
	(typeid(T) == typeid(cuComplex)) ? type = CUDA_C_32F :
	(typeid(T) == typeid(cuDoubleComplex)) ? type = CUDA_C_64F :
	type = CUDA_R_16F;

	return type;
}

template<typename T>
void mat<T>::allocateOnHost(T** h_m, int n_rows, int n_cols, memLocation memLoc) {
	if (memLoc == memLocation::HOST_PINNED) {
		checkCudaErrors(cudaMallocHost((void**)h_m, n_rows * n_cols * sizeof(T)));
	}
	else {
		*h_m = (T*)malloc(n_rows * n_cols * sizeof(T));
	}
}

template<typename T>
void mat<T>::allocateOnDevice(T** d_m, int n_rows, int n_cols) {
	checkCudaErrors(cudaMalloc((void**)d_m, n_rows * n_cols * sizeof(T)));
}

template<typename T>
void mat<T>::emptyAllocation(T** m, int n_rows, int n_cols, memLocation memLoc, bool empty_mat, int* r, int* c) {
	if (empty_mat) {
		*r = n_rows;
		*c = n_cols;
		(HOST_ALLOC(memLoc)) ? allocateOnHost(m, n_rows, n_rows, memLoc) : allocateOnDevice(m, n_rows, n_rows);
	}
}

template<typename T>
void mat<T>::copyHostToDevice(T* h_m, T* d_m, int n_rows, int n_cols) {
	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreate(&stream));
	checkCudaErrors(cudaMemcpyAsync(d_m, h_m, n_rows * n_cols * sizeof(T), cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaStreamSynchronize(stream));
	checkCudaErrors(cudaStreamDestroy(stream));
}

template<typename T>
void mat<T>::copyHostToDevice(const T* h_m, T* d_m, int n_rows, int n_cols) {
	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreate(&stream));
	checkCudaErrors(cudaMemcpyAsync(d_m, h_m, n_rows * n_cols * sizeof(T), cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaStreamSynchronize(stream));
	checkCudaErrors(cudaStreamDestroy(stream));
}

template<typename T>
void mat<T>::copyHostToHost(T* h_m1, T* h_m2, int n_rows, int n_cols) {
	memcpy(h_m2, h_m1, n_rows * n_cols * sizeof(T));
}

template<typename T>
void mat<T>::copyHostToHost(const T* h_m1, T* h_m2, int n_rows, int n_cols) {
	memcpy(h_m2, h_m1, n_rows * n_cols * sizeof(T));
}

template<typename T>
void mat<T>::copyDeviceToHost(T* d_m, T* h_m, int n_rows, int n_cols) {
	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreate(&stream));
	checkCudaErrors(cudaMemcpyAsync(h_m, d_m, n_rows * n_cols * sizeof(T), cudaMemcpyDeviceToHost, stream));
	checkCudaErrors(cudaStreamSynchronize(stream));
	checkCudaErrors(cudaStreamDestroy(stream));
}

template<typename T>
void mat<T>::copyDeviceToHost(const T* d_m, T* h_m, int n_rows, int n_cols) {
	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreate(&stream));
	checkCudaErrors(cudaMemcpyAsync(h_m, d_m, n_rows * n_cols * sizeof(T), cudaMemcpyDeviceToHost, stream));
	checkCudaErrors(cudaStreamDestroy(stream));
}

template<typename T>
void mat<T>::copyDeviceToDevice(T* d_m1, T* d_m2, int n_rows, int n_cols) {
	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreate(&stream));
	checkCudaErrors(cudaMemcpyAsync(d_m2, d_m1, n_rows * n_cols * sizeof(T), cudaMemcpyDeviceToDevice, stream));
	checkCudaErrors(cudaStreamDestroy(stream));
}

template<typename T>
void mat<T>::copyDeviceToDevice(const T* d_m1, T* d_m2, int n_rows, int n_cols) {
	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreate(&stream));
	checkCudaErrors(cudaMemcpyAsync(d_m2, d_m1, n_rows * n_cols * sizeof(T), cudaMemcpyDeviceToDevice, stream));
	checkCudaErrors(cudaStreamDestroy(stream));
}

// copy from mat2 to mat1
template<typename T>
void mat<T>::dynamicCopy(T* data1, T* data2, int n_rows, int n_cols, memLocation memLoc1, memLocation memLoc2) {
	if (HOST_ALLOC(memLoc1) && HOST_ALLOC(memLoc2))
		copyHostToHost(data2, data1, n_rows, n_cols);
	else if (HOST_ALLOC(memLoc1) && DEVICE_ALLOC(memLoc2))
		copyDeviceToHost(data2, data1, n_rows, n_cols);
	else if (DEVICE_ALLOC(memLoc1) && HOST_ALLOC(memLoc2))
		copyHostToDevice(data2, data1, n_rows, n_cols);
	else
		copyDeviceToDevice(data2, data1, n_rows, n_cols);
}

template<typename T>
bool mat<T>::checkToAllocateOnDevice(const T* h_m, T** d_m, int n_rows, int n_cols, memLocation memLoc) {
	if (memLoc == memLocation::HOST || memLoc == memLocation::HOST_PINNED) {
		allocateOnDevice(d_m, n_rows, n_cols);
		copyHostToDevice(h_m, *d_m, n_rows, n_cols);
		return true;
	}
	return false;
}

// wrapper to deal with all data types - overload for all types available
void cublasGeam_wrapper(cublasHandle_t handle,
						cublasOperation_t transa,
						cublasOperation_t transb,
						int m, int n,
						const float* alpha,
						const float* A, int lda,
						const float* beta,
						const float* B, int ldb,
						float* C, int ldc) {

	checkCuBLAS_status(cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
}

void cublasGeam_wrapper(cublasHandle_t handle,
						cublasOperation_t transa,
						cublasOperation_t transb,
						int m, int n,
						const double* alpha,
						const double* A, int lda,
						const double* beta,
						const double* B, int ldb,
						double* C, int ldc) {

	checkCuBLAS_status(cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
}

void cublasGeam_wrapper(cublasHandle_t handle,
						cublasOperation_t transa,
						cublasOperation_t transb,
						int m, int n,
						const cuComplex* alpha,
						const cuComplex* A, int lda,
						const cuComplex* beta,
						const cuComplex* B, int ldb,
						cuComplex* C, int ldc) {

	checkCuBLAS_status(cublasCgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
}

void cublasGeam_wrapper(cublasHandle_t handle,
						cublasOperation_t transa,
						cublasOperation_t transb,
						int m, int n,
						const cuDoubleComplex* alpha,
						const cuDoubleComplex* A, int lda,
						const cuDoubleComplex* beta,
						const cuDoubleComplex* B, int ldb,
						cuDoubleComplex* C, int ldc) {

	checkCuBLAS_status(cublasZgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
}


// C is defined to be allocated on device in matrix operators - no need for
// memory Location checks
template<typename T>
void mat<T>::matAdd(cublasOperation_t transa,
					cublasOperation_t transb,
					int m, int n,
					const T** A, int lda,
					memLocation Aloc,
					const T** B, int ldb,
					memLocation Bloc,
					T** C, int ldc,
					memLocation Cloc) {

	const T alpha = static_cast<T>(1);
	const T beta = static_cast<T>(1);
	cublasHandle_t handle;


	// copy host data to device if needed
	T* d_A = NULL; T* d_B = NULL; bool A_status; bool B_status;
	A_status = checkToAllocateOnDevice(*A, &d_A, lda, n, Aloc);
	B_status = checkToAllocateOnDevice(*B, &d_B, ldb, n, Bloc);

	// create handle
	checkCuBLAS_status(cublasCreate_v2(&handle));

	// compute
	cublasGeam_wrapper(handle,
					   transa,
					   transb,
					   m, n,
					   &alpha,
					   (A_status) ? d_A : *A,
					   lda,
					   &beta,
					   (B_status) ? d_B : *B,
					   ldb,
					   *C,
					   ldc);

	checkCudaErrors(cudaDeviceSynchronize());

	// free memory from GPU
	if (A_status)checkCudaErrors(cudaFree(d_A));
	if (B_status)checkCudaErrors(cudaFree(d_B));

	// destroy handle
	checkCuBLAS_status(cublasDestroy_v2(handle));
}

// matrix multiplication wrapper to deal with all data types
template<typename T>
void cublasGemm_wrapper(cublasHandle_t handle,
						cublasOperation_t transa,
						cublasOperation_t transb,
						int m, int n, int k,
						const T* alpha,
						const T* A,
						cudaDataType Atype,
						int lda,
						const T* B,
						cudaDataType Btype,
						int ldb,
						const T* beta,
						T* C,
						cudaDataType Ctype,
						int ldc,
						cudaDataType computeType,
						cublasGemmAlgo_t algo) {

	checkCuBLAS_status(cublasGemmEx(handle,
									transa,
									transb,
									m, n, k,
									alpha,
									A, Atype, lda,
									B, Btype, ldb,
									beta,
									C, Ctype, ldc,
									computeType,
									algo));
}

// C is defined to be allocated on device in matrix operators - no need for
// memory Location check
template<typename T>
void mat<T>::matMul(cublasOperation_t transa,
					cublasOperation_t transb,
					int m, int n, int k,
					const T** A,
					cudaDataType Atype,
					memLocation Aloc,
					int lda,
					const T** B,
					cudaDataType Btype,
					memLocation Bloc,
					int ldb,
					T** C,
					cudaDataType Ctype,
					int ldc,
					cudaDataType computeType,
					cublasGemmAlgo_t algo) {

	const T alpha = static_cast<T>(1);
	const T beta = static_cast<T>(0);
	cublasHandle_t handle;

	// copy host data to device if needed
	T* d_A = nullptr; T* d_B = nullptr; bool A_status; bool B_status;
	A_status = checkToAllocateOnDevice(*A, &d_A, lda, n, Aloc);
	B_status = checkToAllocateOnDevice(*B, &d_B, ldb, n, Bloc);

	// create handle
	checkCuBLAS_status(cublasCreate_v2(&handle));

	// compute
	cublasGemm_wrapper(handle,
					   transa,
					   transb,
					   m, n, k,
					   &alpha,
					   (A_status) ? d_A : *A,
					   Atype, lda,
					   (B_status) ? d_B : *B,
					   Btype, ldb,
					   &beta,
					   *C,
					   Ctype, ldc,
					   computeType,
					   algo);

	checkCudaErrors(cudaDeviceSynchronize());

	// free memory from GPU
	if (A_status)checkCudaErrors(cudaFree(d_A));
	if (B_status)checkCudaErrors(cudaFree(d_B));

	// destroy handle
	checkCuBLAS_status(cublasDestroy_v2(handle));
}

template<typename T>
void mat<T>::destroyHostMatrix(T** h_m, memLocation memLoc) {
	if (memLoc == memLocation::HOST_PINNED) checkCudaErrors(cudaFreeHost(*h_m));
	else free(*h_m);
}

template<typename T>
void mat<T>::destroyDeviceMatrix(T** d_m) {
	checkCudaErrors(cudaFree(*d_m));
}

// calculating matrix transpose using shared memory and a tiled algorithm
/*
the idea is to minimize the time of jumps in device memory (it is much slower in GPU than CPU)
through jumping between close neighberhoods of data items in memory ONLY. 
optimizations:
1) shared memory - the transpose operation requires many threads to read from the same matrix,
   and to write to the same output result, therefore using shared memory (duplicating the same memory 
   for each thread) results in much faster read/write operations.

2) tile divition of a matrix - divide the matrix into tiles of kxk (for A(NxN), N | k), performe the
   transpose of each tile and copy the transposed tiles to the output matrix. this methods minimizes
   long jumps in physical memory when used with shared memory.
*/
// execution in blocks of [BLOCK_ROWS, TILE_DIM]
template<typename T>
__global__ void tiledTranspose(T* idata, T* odata) {
	// running two dimensions - block dimension is TILE_DIM
	int x = TILE_DIM * blockIdx.x + threadIdx.x;
	int y = BLOCK_ROWS * blockIdx.y + threadIdx.y;
	int height = gridDim.y * TILE_DIM;

	// allocating tile in shared memory - preventing bank conflicts
	__shared__ T tile[TILE_DIM][TILE_DIM + 1];

	// copying idata to tile
	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
		tile[threadIdx.y + j][threadIdx.x] = idata[x * height + y + j];
	}

	__syncthreads();

	// flip indices
	x = TILE_DIM * blockIdx.y + threadIdx.x;
	y = TILE_DIM * blockIdx.x + threadIdx.y;

	// copy tile data to odata in flipped order
	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
		odata[x * height + y + j] = tile[threadIdx.x][threadIdx.y + j];
	}
}

template<typename T>
void mat<T>::transpose() {
	// check if data is in device memory
	T* idata; T* odata; 
	bool i_st = checkToAllocateOnDevice(data, &idata, M, N, memState);
	checkCudaErrors(cudaMalloc((void**)&odata, N * M * sizeof(T)));

	// check for fitting dimensions
	if (N % TILE_DIM || M % BLOCK_ROWS) throw std::invalid_argument("tile dimensions have to divide matrix dimensions");

	// compute transpose kernel
	dim3 threadsPerBlock(TILE_DIM, BLOCK_ROWS);
	dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
	tiledTranspose<T> <<<numBlocks,threadsPerBlock>>> ((i_st) ? idata : data, odata);

	// copy memory to host
	int new_dim_x = N; 
	int new_dim_y = M;
	M = new_dim_x; N = new_dim_y;
	checkCudaErrors(cudaMemcpy(data, odata, N * M * sizeof(T), cudaMemcpyDeviceToHost));
	// free memory
	if (i_st)checkCudaErrors(cudaFree(idata));
	checkCudaErrors(cudaFree(odata));
}

/*
---------------------------------------------------------------------------------------------------
-----------------------------------SPARSE MATRIX CLASS (DERIVED)-----------------------------------
---------------------------------------------------------------------------------------------------
*/

template<typename T>
class Sparse_mat : public mat<T> {
public:
	/*
	CSR data - with sparse-dense multiplications the user can ignore the CSR format and use only the cuSparseLt api.
	in that case, the structured matrix is brought from a dense encoding (regular pointer to array of size MxN
	in column major order) to a pruned compressed format, then the matrix is multiplied with the dense right/left
	size matrix. with sparse-sparse operations the api uses the cuda tool cuSPARSE which demands a specific
	encoded format for our sparse matrix, hence the need for CSR.
	*/
	int32_t* csrRowOffsets; // size (M + 1) x 1 - encoding of row indices of nz elements
	int32_t* csrColInd; // size nnz x 1 - column indices of nz elements
	T* csrValues; // size nnz x 1 - values of all nz elements
	int32_t nnz;

	// flag for detecting creation of CSR format
	bool CSR_enabled = false;

	// needed for creating csr formatting and for sparse - sparse operations
	cusparseSpMatDescr_t SpMatDescr;

	// handling CSR format for sparse - sparse operations in cuSPARSE
	// includes creating a sparseMatrix handle (in createCSR) and destroying it (in destroyCSR)
	void createCSR();
	void destroyCSR();

	// inherit constructors
	using mat<T>::mat;

	// sparse on dense
	Sparse_mat<T> operator*(const mat<T>& B);
	// sparse on sparse
	Sparse_mat<T> operator*(const Sparse_mat<T>& B);
	// sparse on dense
	Sparse_mat<T>& operator*=(const mat<T>& B);
	// sparse on sparse
	Sparse_mat<T>& operator*=(const Sparse_mat<T>& B);

private:
	/*
	---------------- STRUCTURED SPARSE - DENSE MULTIPLICATION USING cuSPARSELt ---------------- 
	*/

	// calculate compute type for cusparse and cusparseLt
	cusparseComputeType cusparseLtC_T();
	cudaDataType cusparseM_T();

	// initialize descriptors for structured matrix settings (for sparse_mat<T> : mat<T>)
	// and for dense matrix settings (for mat<T>)
	void cusparseLtStructuredMatInit(const cusparseLtHandle_t* handle,
									 cusparseLtMatDescriptor_t* matDescr,
									 int64_t rows,
									 int64_t cols,
		                             int64_t ld,
		                             uint32_t alignment,
		                             cudaDataType valueType,
		                             cusparseLtSparsity_t sparsity);

	void cusparseLtDenseMatInit(const cusparseLtHandle_t* handle,
								cusparseLtMatDescriptor_t* matDescr,
								int64_t rows,
								int64_t cols,
								int64_t ld,
								uint32_t alignment,
								cudaDataType valueType);

	// set and get attributes for number of batchs and strides for batched multiplication
	void cusparseLtSetBatchAttribues(const cusparseLtHandle_t* handle,
									 cusparseLtMatDescriptor_t* matmulDescr,
									 const void* num_batches,
									 const void* stride,
									 size_t batchDataSize,
									 size_t strideDataSize);

	// set mat-mul descriptor initialization
	void cusparseLtMatMulInit(const cusparseLtHandle_t* handle,
							  cusparseLtMatmulDescriptor_t* matmulDescr,
							  cusparseOperation_t opA,
							  cusparseOperation_t opB,
							  const cusparseLtMatDescriptor_t* matA,
							  const cusparseLtMatDescriptor_t* matB,
							  const cusparseLtMatDescriptor_t* matC,
							  const cusparseLtMatDescriptor_t* matD);

	// initialize and set algorithm for mat-mul computation
	void cusparseLtAlgSelectInit(const cusparseLtHandle_t* handle,
								 cusparseLtMatmulAlgSelection_t* algSelection,
								 const cusparseLtMatmulDescriptor_t* matmulDescr,
								 cusparseLtMatmulAlg_t alg);

	// plan mat-mul computation
	void cusparseLtPlanInit(const cusparseLtHandle_t* handle,
							cusparseLtMatmulPlan_t* plan,
							const cusparseLtMatmulDescriptor_t* matmulDescr,
							const cusparseLtMatmulAlgSelection_t* algSelection);

	// unite all initializers in one main function
	void cusparseLtInitDescriptors(const cusparseLtHandle_t** handle,
								   cusparseLtMatDescriptor_t matDescrs[],
								   int64_t a_rows, int64_t a_cols,
								   int64_t b_rows, int64_t b_cols,
								   int64_t c_rows, int64_t c_cols,
								   cudaDataType valueType,
								   cusparseLtMatmulDescriptor_t** matmulDescr,
								   cusparseLtMatmulAlgSelection_t** algSelection,
								   cusparseLtMatmulPlan_t** plan);

	// prune and compress structured (sparse) matrices
	void cusparseLtSparseMatPrune(const cusparseLtHandle_t* handle,
								  const cusparseLtMatmulDescriptor_t* matmulDescr,
								  const void* d_in,
		                          void** d_out,
		                          cudaStream_t stream);

	void cusparseLtSparseMatCompress(const cusparseLtHandle_t* handle,
									 const cusparseLtMatmulPlan_t* plan,
									 const void* d_dense,
									 void** d_compressed,
									 cudaStream_t stream);

	// prune and compress structured matrix
	void cusparseLtSparseMatPrune_Compress(const cusparseLtHandle_t* handle,
										   const cusparseLtMatmulDescriptor_t* matmulDescr,
										   const cusparseLtMatmulPlan_t* plan,
										   const T* denseMat,
										   int sizeA,
										   memLocation Loc,
										   void** d_compressed,
										   cudaStream_t stream);

	// search for optimal execution kernel
	void cusparseLtSearchOptimalKernel(const cusparseLtHandle_t* handle,
									   cusparseLtMatmulPlan_t* plan,
									   const void* alpha,
									   const void* d_A,
									   const void* d_B,
									   const void* beta,
									   const void* d_C,
									   void** d_D,
									   void* workspace,
									   cudaStream_t* streams,
									   int32_t numStreams);

	// execute operation
	void structuredDenseMatMul(const cusparseLtHandle_t* handle,
						   cusparseLtMatmulPlan_t* plan,
						   int sizeA, int sizeB, int sizeC,
						   const T* A,
						   memLocation Aloc,
						   const T* B,
						   memLocation Bloc,
						   const T* C,
						   memLocation Cloc,
						   T** d_D,
						   void* workspace,
						   cudaStream_t* streams,
						   int32_t numStreams);

	// execute matrix multiplication - entire process
	void structuredDenseMatMul_op(int64_t b_rows, int64_t b_cols,
								  const T* B,
								  memLocation Bloc,
								  int64_t c_rows, int64_t c_cols,
								  const T* C,
								  memLocation Cloc,
								  T** D);
	/*
	---------------- SPARSE - SPARSE MULTIPLICATION USING cuSPARSE ----------------
	*/

	// create a dense representation
	void createDenseFormat(cusparseDnMatDescr_t* dnMatDescr,
						   int32_t rows,
						   int32_t cols,
						   int32_t ld,
						   const T* values,
						   T** device_values);

	// overload - for empty initialization (used for copying data from sparse/dense formats to 
	// current matrix dense format)
	void createDenseFormat(cusparseDnMatDescr_t* dnMatDescr,
						   int32_t rows,
						   int32_t cols,
						   int32_t ld,
						   T* device_values);

	// create the CSR format in the sparse matrix descriptor
	void createCsrSparse(cusparseSpMatDescr_t* spMatDescr,
						 int32_t rows,
						 int32_t cols,
						 int32_t nnz,
						 void** csrRowOffsets,
						 void** csrColInd,
						 void** csrValues);

	// allocate and set csr format arrays
	void setCsrArrays(cusparseSpMatDescr_t* spMatDescr,
					  int32_t rowsSize,
					  int32_t nnz,
					  int32_t** d_csrRowsOffsets,
					  bool rowsAlloc,
					  int32_t** d_csrColInd,
					  bool colsAlloc,
					  T** d_csrValues,
					  bool valuesAlloc);

	// convert the dense format of a matrix to a sparse CSR format, allocate CSR arrays in device
	// create a dense matrix and convert it to a sparse CSR representation
	void convertDenseToCsrFormat(cusparseHandle_t handle,
								 cusparseSpMatDescr_t* matB,
								 int32_t rows,
								 int32_t cols,
								 int32_t ld,
								 const T* values,
								 memLocation memLoc,
								 int32_t** d_csrColInd,
								 T** d_csrValues,
		                         int32_t** d_csrRowOffsets,
								 int32_t* d_nnz);

	// get first buffer for spGEMM computation 
	void getWorkEstimation_allocBuffers(cusparseHandle_t handle,
										const void* alpha,
										cusparseSpMatDescr_t matA,
										cusparseSpMatDescr_t matB,
										const void* beta,
										cusparseSpMatDescr_t matC,
										cudaDataType computeType,
										cusparseSpGEMMDescr_t spgemmDescr,
										size_t* bufferSize1,
										void** externalBuffer1);

	// get second buffer and do the actual computation for sparse-sparse matmul
	void computeSpGEMM_allocBuffers(cusparseHandle_t handle,
									const void* alpha,
									cusparseSpMatDescr_t matA,
									cusparseSpMatDescr_t matB,
									const void* beta,
									cusparseSpMatDescr_t matC,
									cudaDataType computeType,
									cusparseSpGEMMDescr_t spgemmDescr,
									size_t* bufferSize2,
									void** externalBuffer2);

	// get workspace and compute
	void spGEMM(cusparseHandle_t handle,
				const void* alpha,
				cusparseSpMatDescr_t matA,
				cusparseSpMatDescr_t matB,
				const void* beta,
				cusparseSpMatDescr_t matC,
				cudaDataType computeType,
				cusparseSpGEMMDescr_t spgemmDescr,
				size_t* bufferSize1,
				void** externalBuffer1,
				size_t* bufferSize2,
				void** externalBuffer2);

	// copy from descriptor to actual matrix
	void copySpGEMM_toMat(cusparseHandle_t handle,
						  const void* alpha,
						  cusparseSpMatDescr_t matA,
						  cusparseSpMatDescr_t matB,
						  const void* beta,
						  cusparseSpMatDescr_t matC,
						  cudaDataType computeType,
						  cusparseSpGEMMDescr_t spgemmDescr);

	// sparse to sparse matmul (cusparse standard)
	Sparse_mat<T> sparseSparseMatMul(cusparseSpMatDescr_t matB, int32_t B_cols);

	// get CSR data from sparse matrix descriptor (if needed)
	void getCSR_data(int64_t* rows,
					 int64_t* cols,
					 int64_t* nnz,
					 void** csrRowOffsets,
					 void** csrColInd,
					 void** csrValues,
					 cusparseIndexType_t* csrRowOffsetsType,
					 cusparseIndexType_t* csrColIndType);

	// convert from CSR format to a dense matrix format 
	void convertCSRformatToDense(cusparseSpMatDescr_t matA, T** values, int rows, int cols, int ld);
};

// create CSR formatted matrix 
// used when a matrix has known dimensions and values, for sparse-sparse cuSPARSE api operations only
template<typename T>
void Sparse_mat<T>::createCSR() {
	// create handle, updating flag
	cusparseHandle_t handle; CSR_enabled = true;
	checkCuSparseErrors(cusparseCreate(&handle));
	// allocate memory for the three arrays and convert from a dense representation
	// to a CSR sparse matrix representation
	convertDenseToCsrFormat(handle,
							&SpMatDescr,
							(int32_t)M,
							(int32_t)N,
							(int32_t)M,
							(const T*)data,
							memState,
							&csrColInd,
							&csrValues,
							&csrRowOffsets,
							&nnz);
	// destroy handle
	checkCuSparseErrors(cusparseDestroy(handle));
}

// destroy CSR format sparse matrix handle and free all csr arrays
template<typename T>
void Sparse_mat<T>::destroyCSR() {
	cudaFreeMem((void*)csrRowOffsets, (void*)csrColInd, (void*)csrValues);
	if(CSR_enabled)checkCuSparseErrors(cusparseDestroySpMat(SpMatDescr));
	CSR_enabled = false;
}

// determine compute type for cusparseLt sparse-dense multiplication routine
template<typename T>
cusparseComputeType Sparse_mat<T>::cusparseLtC_T() {
	cusparseComputeType type;
	(typeid(T) == typeid(float)) ? type = CUSPARSE_COMPUTE_TF32_FAST :
	(typeid(T) == typeid(__half)) ? type = CUSPARSE_COMPUTE_16F :
	(typeid(T) == typeid(int)) ? type = CUSPARSE_COMPUTE_32I :
	type = CUSPARSE_COMPUTE_16F; // for F16B
	return type;
}

template<typename T>
cudaDataType Sparse_mat<T>::cusparseM_T() {
	// type to be returned
	cudaDataType type;

	(typeid(T) == typeid(float)) ? type = CUDA_R_32F :
	(typeid(T) == typeid(double)) ? type = CUDA_R_64F :
	(typeid(T) == typeid(cuComplex)) ? type = CUDA_C_32F :
	(typeid(T) == typeid(cuDoubleComplex)) ? type = CUDA_C_64F :
	type = CUDA_R_16F;

	return type;
}

/*
--------------------------------- operation D = alpha*op(A)*op(B) + beta*C ---------------------------------
A is sparse, B , C are dense
*/

/*
wrapper for structured matrix descriptor init function - structured refers to matrices
with specific structures that allow lower computation complexity compared to dense matrices.
for example : diagonal matrices, general sparse matrices, jordan block form, lines around diagonal etc.
*/
template<typename T>
void Sparse_mat<T>::cusparseLtStructuredMatInit(const cusparseLtHandle_t* handle,
												cusparseLtMatDescriptor_t* matDescr,
												int64_t rows,
												int64_t cols,
												int64_t ld,
												uint32_t alignment,
												cudaDataType valueType,
												cusparseLtSparsity_t sparsity) {

	checkCuSparseErrors(cusparseLtStructuredDescriptorInit(handle,
														   matDescr,
														   rows, cols,
														   ld,
														   alignment,
														   valueType,
														   CUSPARSE_ORDER_COL,
														   sparsity));
}

/*
wrapper for dense matrix descriptor init function - initializes the matDescr 
*/
template<typename T>
void Sparse_mat<T>::cusparseLtDenseMatInit(const cusparseLtHandle_t* handle,
										   cusparseLtMatDescriptor_t* matDescr,
										   int64_t rows,
									       int64_t cols,
										   int64_t ld,
										   uint32_t alignment,
										   cudaDataType valueType) {

	checkCuSparseErrors(cusparseLtDenseDescriptorInit(handle,
													  matDescr,
												      rows, cols,
													  ld,
													  alignment,
													  valueType,
													  CUSPARSE_ORDER_COL));
}

// set the number of batches and stride for batched operation with a stream per batch
template<typename T>
void Sparse_mat<T>::cusparseLtSetBatchAttribues(const cusparseLtHandle_t* handle,
												cusparseLtMatDescriptor_t* matmulDescr,
												const void* num_batches,
												const void* stride,
												size_t batchDataSize,
												size_t strideDataSize) {

	checkCuSparseErrors(cusparseLtMatDescSetAttribute(handle,
													  matmulDescr,
													  CUSPARSELT_MAT_NUM_BATCHES,
													  num_batches,
													  batchDataSize));

	checkCuSparseErrors(cusparseLtMatDescSetAttribute(handle,
													  matmulDescr,
													  CUSPARSELT_MAT_BATCH_STRIDE,
													  stride,
													  strideDataSize));
}

// calculate compute type for matmul


// set matmul descriptor with all matrices data - this a descriptor for a sparse-dense multiplication
template<typename T>
void Sparse_mat<T>::cusparseLtMatMulInit(const cusparseLtHandle_t* handle,
										 cusparseLtMatmulDescriptor_t* matmulDescr,
										 cusparseOperation_t opA,
										 cusparseOperation_t opB,
										 const cusparseLtMatDescriptor_t* matA,
										 const cusparseLtMatDescriptor_t* matB,
										 const cusparseLtMatDescriptor_t* matC,
										 const cusparseLtMatDescriptor_t* matD) {

	auto computeType = cusparseLtC_T();
	checkCuSparseErrors(cusparseLtMatmulDescriptorInit(handle,
													   matmulDescr,
													   opA, opB,
													   matA, matB,
													   matC, matD,
													   computeType));
}

// select multiplication algorithm
template<typename T>
void Sparse_mat<T>::cusparseLtAlgSelectInit(const cusparseLtHandle_t* handle,
											cusparseLtMatmulAlgSelection_t* algSelection,
											const cusparseLtMatmulDescriptor_t* matmulDescr,
											cusparseLtMatmulAlg_t alg) {

	checkCuSparseErrors(cusparseLtMatmulAlgSelectionInit(handle, algSelection, matmulDescr, alg));
}

// plan the multiplication operation
template<typename T>
void Sparse_mat<T>::cusparseLtPlanInit(const cusparseLtHandle_t* handle,
									   cusparseLtMatmulPlan_t* plan,
									   const cusparseLtMatmulDescriptor_t* matmulDescr,
									   const cusparseLtMatmulAlgSelection_t* algSelection) {

	size_t workspaceSize;
	// get workspace size in bytes
	checkCuSparseErrors(cusparseLtMatmulGetWorkspace(handle, algSelection, &workspaceSize));
	checkCuSparseErrors(cusparseLtMatmulPlanInit(handle, plan, matmulDescr, algSelection, workspaceSize));
}

// initialize all needed descriptors for matmul routine
template<typename T>
void Sparse_mat<T>::cusparseLtInitDescriptors(const cusparseLtHandle_t** handle,
											  cusparseLtMatDescriptor_t matDescrs[],
											  int64_t a_rows, int64_t a_cols,
											  int64_t b_rows, int64_t b_cols,
											  int64_t c_rows, int64_t c_cols,
											  cudaDataType valueType,
											  cusparseLtMatmulDescriptor_t** matmulDescr,
											  cusparseLtMatmulAlgSelection_t** algSelection,
											  cusparseLtMatmulPlan_t** plan) {

	// init A
	cusparseLtStructuredMatInit(*handle,
								&matDescrs[0],
								a_rows, a_cols,
								a_rows,
								16,
								valueType,
								CUSPARSELT_SPARSITY_50_PERCENT);

	// init B, C, D - cuSparseLt supports only D with identical dimensions to C
	cusparseLtDenseMatInit(*handle, &matDescrs[1], b_rows, b_cols, b_rows, 16, valueType);
	cusparseLtDenseMatInit(*handle, &matDescrs[2], c_rows, c_cols, c_rows, 16, valueType);
	cusparseLtDenseMatInit(*handle, &matDescrs[3], c_rows, c_cols, c_rows, 16, valueType);

	// initialize matmul descriptor
	cusparseLtMatMulInit(*handle,
		*matmulDescr,
		CUSPARSE_OPERATION_NON_TRANSPOSE,
		CUSPARSE_OPERATION_NON_TRANSPOSE,
		&matDescrs[0], &matDescrs[1], &matDescrs[2], &matDescrs[3]);
	// initialize algorithm selection
	cusparseLtAlgSelectInit(*handle, *algSelection, *matmulDescr, CUSPARSELT_MATMUL_ALG_DEFAULT);
	// initialize plan
	cusparseLtPlanInit(*handle, *plan, *matmulDescr, *algSelection);
}

// prune structured matrix
template<typename T>
void Sparse_mat<T>::cusparseLtSparseMatPrune(const cusparseLtHandle_t* handle,
											 const cusparseLtMatmulDescriptor_t* matmulDescr,
											 const void* d_in,
											 void** d_out,
											 cudaStream_t stream) {

	int* d_valid; int isValid;
	// malloc result in memory
	checkCudaErrors(cudaMalloc((void**)&d_valid, sizeof(d_valid)));

	checkCuSparseErrors(cusparseLtSpMMAPrune(handle, matmulDescr, d_in, *d_out, CUSPARSELT_PRUNE_SPMMA_TILE, stream));
	checkCuSparseErrors(cusparseLtSpMMAPruneCheck(handle, matmulDescr, *d_out, d_valid, stream));

	// copy to host integer
	checkCudaErrors(cudaMemcpyAsync(&isValid, d_valid, sizeof(int), cudaMemcpyDeviceToHost, stream));
	checkCudaErrors(cudaStreamSynchronize(stream));
	// free d_valid
	checkCudaErrors(cudaFree(d_valid));

	// check for prune success
	if (isValid) {
		throw std::runtime_error("!!!!The matrix has been pruned in a wrong way.cusparseLtMatmul will not provide correct results");
	}
}

// compress pruned matrix
template<typename T>
void Sparse_mat<T>::cusparseLtSparseMatCompress(const cusparseLtHandle_t* handle,
												const cusparseLtMatmulPlan_t* plan,
												const void* d_dense,
												void** d_compressed,
												cudaStream_t stream) {

	size_t compressedSize;
	checkCuSparseErrors(cusparseLtSpMMACompressedSize(handle, plan, &compressedSize));
	// allocate compressedSize bytes for d_compressed
	checkCudaErrors(cudaMalloc(d_compressed, compressedSize)); // remember to free at the end of the program
	checkCuSparseErrors(cusparseLtSpMMACompress(handle, plan, d_dense, *d_compressed, stream));
}

// prune and compress the structured matrix
template<typename T>
void Sparse_mat<T>::cusparseLtSparseMatPrune_Compress(const cusparseLtHandle_t* handle,
													  const cusparseLtMatmulDescriptor_t* matmulDescr,
													  const cusparseLtMatmulPlan_t* plan,
													  const T* denseMat,
													  int sizeA,
													  memLocation Loc,
													  void** d_compressed,
													  cudaStream_t stream) {

	// check if denseMat is in device memory
	T* d_dense;
	bool status = checkToAllocateOnDevice(denseMat, &d_dense, sizeA, 1, Loc);

	// allocate d_out
	T* d_out;
	checkCudaErrors(cudaMalloc((void**)&d_out, (size_t)(sizeA) * sizeof(T)));

	// prune
	cusparseLtSparseMatPrune(handle,
							 matmulDescr,
							 (status) ? (const void*)d_dense : (const void*)denseMat,
							 (void**)&d_out,
							 stream);

	// compress
	cusparseLtSparseMatCompress(handle, plan, d_out, d_compressed, stream);

	// free memory
	if(status)checkCudaErrors(cudaFree(d_dense));
	checkCudaErrors(cudaFree(d_out));
}

// search for fastest algorithm and automatically updates the plan to the optimal multiplication algorithm
template<typename T>
void Sparse_mat<T>::cusparseLtSearchOptimalKernel(const cusparseLtHandle_t* handle,
												  cusparseLtMatmulPlan_t* plan,
												  const void* alpha,
												  const void* d_A,
												  const void* d_B,
												  const void* beta,
												  const void* d_C,
												  void** d_D,
												  void* workspace,
												  cudaStream_t* streams,
												  int32_t numStreams) {

	checkCuSparseErrors(cusparseLtMatmulSearch(handle,
											   plan,
											   alpha,
											   d_A, d_B,
											   beta,
											   d_C, *d_D,
											   workspace,
											   streams,
											   numStreams));
}

// execute sparse-dense matrix multiplication
template<typename T>
void Sparse_mat<T>::structuredDenseMatMul(const cusparseLtHandle_t* handle,
								      cusparseLtMatmulPlan_t* plan,
									  int sizeA, int sizeB, int sizeC,
								      const T* A,
								      memLocation Aloc,
								      const T* B,
								      memLocation Bloc,
								      const T* C,
								      memLocation Cloc,
								      T** d_D,
								      void* workspace,
								      cudaStream_t* streams,
								      int32_t numStreams) {

	const T alpha = static_cast<const T>(1);
	const T beta = static_cast<const T>(0);
	
	// checking for device allocations - the dimensions dont metter for memory allocation - just the size
	T* d_A; T* d_B; T* d_C;
	bool A_status = checkToAllocateOnDevice(A, &d_A, sizeA, 1, Aloc);
	bool B_status = checkToAllocateOnDevice(B, &d_B, sizeB, 1, Bloc);
	bool C_status = checkToAllocateOnDevice(C, &d_C, sizeC, 1, Cloc);

	// search for optimal algorithm
	cusparseLtSearchOptimalKernel(handle, plan,
								  &alpha,
								  (A_status) ? d_A : A,
								  (B_status) ? d_B : B,
								  &beta,
								  (C_status) ? d_C : C,
								  (void**)d_D,
								  workspace,
								  streams,
								  numStreams);

	// perform matrix multiplication
	checkCuSparseErrors(cusparseLtMatmul(handle, 
										 (const cusparseLtMatmulPlan_t*)plan,
										 &alpha,
										 (A_status) ? d_A : A,
										 (B_status) ? d_B : B,
										 &beta,
										 (C_status) ? d_C : C,
										 *d_D,
										 workspace,
										 streams,
										 numStreams));
	
	// free memory if neccessary
	if (A_status)checkCudaErrors(cudaFree(d_A));
	if (B_status)checkCudaErrors(cudaFree(d_B));
	if (C_status)checkCudaErrors(cudaFree(d_C));
}


void destroyMatDescrArr(cusparseLtMatDescriptor_t arr[], int N) {
	for (auto pArr = arr; pArr != arr + N; ++pArr) {
		checkCuSparseErrors(cusparseLtMatDescriptorDestroy(pArr));
	}
}

// sparse - dense multiplication - matrices C, D must be allocated in device memory
template<typename T>
void Sparse_mat<T>::structuredDenseMatMul_op(int64_t b_rows, int64_t b_cols,
											 const T* B,
											 memLocation Bloc,
											 int64_t c_rows, int64_t c_cols,
											 const T* C, 
											 memLocation Cloc,
											 T** D) {

	// init value type
	auto valueType = cusparseM_T();
	// create handle and descriptors
	cusparseLtHandle_t handle; 
	cusparseLtMatDescriptor_t matDescrs[4]; // m[0] = matA, m[1] = matB, m[2] = matC, m[3] = matD
	cusparseLtMatmulDescriptor_t matmulDescr; 
	cusparseLtMatmulAlgSelection_t AlgSel;
	cusparseLtMatmulPlan_t plan;

	// create pointers to descriptors for initialization
	auto pHandle = &handle; auto pMatmulDescr = &matmulDescr;
	auto pAlgSel = &AlgSel; auto pPlan = &plan;

	checkCuSparseErrors(cusparseLtInit(pHandle));
	printf_s("\ngot to multiplying with cuSparseLt\n");
	// initialize all descriptors
	cusparseLtInitDescriptors((const cusparseLtHandle_t**)&pHandle,
							  matDescrs,
							  M, N,
							  b_rows, b_cols,
							  M,
							  b_cols,
							  valueType,
							  &pMatmulDescr,
							  &pAlgSel,
							  &pPlan);

	// create stream
	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreate(&stream));
	// prune and compress matrix A (the sparse matrix)
	T* d_A_cmprs;
	cusparseLtSparseMatPrune_Compress(pHandle, pMatmulDescr, pPlan, (const T*)data, M * N, memState, (void**)&d_A_cmprs, stream);

	// search for optimal algorithm and execute operation
	structuredDenseMatMul((const cusparseLtHandle_t*)pHandle,
						  pPlan,
						  (int)(M * N), 
						  (int)(b_rows * b_cols),
						  (int)(M * b_cols),
						  (const T*)d_A_cmprs,
						  memLocation::DEVICE,
						  (const T*)B,
						  Bloc,
						  (const T*)C,
						  Cloc, 
						  D, 
						  nullptr, 
						  &stream, 1);

	// free memory
	checkCudaErrors(cudaStreamDestroy(stream));
	checkCudaErrors(cudaFree(d_A_cmprs));
	destroyMatDescrArr(matDescrs, 4);
	checkCuSparseErrors(cusparseLtMatmulPlanDestroy(pPlan));
	checkCuSparseErrors(cusparseLtDestroy(&handle));
}

/*
--------------------------------- operation C = alpha*op(A)*op(B) + beta*C ---------------------------------
A, B are sparse
*/

/*
convert the current matrix representation into a CSR (un)compressed format of a sparse matrix
the matrix is converted into a dense matrix representation and then converted into a CSR format
*/

// convert to dense matrix format
template<typename T>
void Sparse_mat<T>::createDenseFormat(cusparseDnMatDescr_t* dnMatDescr,
									  int32_t rows,
									  int32_t cols,
	                                  int32_t ld,
									  const T* values,
									  T** device_values) {

	// check if the matrix is in device or not
	bool Dense_status = checkToAllocateOnDevice(values, device_values, (int)rows, (int)cols, memState);

	// create dense matrix handle
	cudaDataType valueType = cusparseM_T();
	checkCuSparseErrors(cusparseCreateDnMat(dnMatDescr,
											rows,
											cols,
											ld,
											(Dense_status) ? (void*)*device_values : (void*)values,
											valueType,
											CUSPARSE_ORDER_COL));
}

// initialize dense matrix descriptor
template<typename T>
void Sparse_mat<T>::createDenseFormat(cusparseDnMatDescr_t* dnMatDescr,
									  int32_t rows,
									  int32_t cols,
									  int32_t ld,
									  T* device_values) {

	// create dense matrix handle
	cudaDataType valueType = cusparseM_T();
	checkCuSparseErrors(cusparseCreateDnMat(dnMatDescr,
											rows,
											cols,
											ld,
											(void*)device_values,
											valueType,
											CUSPARSE_ORDER_COL));
}

// create a csr formatted sparse matrix representation
template<typename T>
void Sparse_mat<T>::createCsrSparse(cusparseSpMatDescr_t* spMatDescr,
								    int32_t rows,
								    int32_t cols,
								    int32_t nnz,
								    void** csrRowOffsets,
								    void** csrColInd,
								    void** csrValues) {

	cudaDataType valueType = cusparseM_T();
	checkCuSparseErrors(cusparseCreateCsr(spMatDescr,
										  rows, cols,
										  nnz,
										  *csrRowOffsets,
										  *csrColInd,
										  *csrValues,
										  CUSPARSE_INDEX_32I,
										  CUSPARSE_INDEX_32I,
										  CUSPARSE_INDEX_BASE_ZERO,
										  valueType));
}

// allocate buffer for convertion process
void alloc_analyze_convertBuffer(cusparseHandle_t handle,
								 cusparseDnMatDescr_t matA,
								 cusparseSpMatDescr_t matB,
								 cusparseDenseToSparseAlg_t alg,
								 size_t* bufferSize,
								 void** dBuffer) {

	checkCuSparseErrors(cusparseDenseToSparse_bufferSize(handle, matA, matB, alg, bufferSize));
	checkCudaErrors(cudaMalloc(dBuffer, *bufferSize)); // REMEMBER TO FREE

	// update nnz in the descriptor of matB
	checkCuSparseErrors(cusparseDenseToSparse_analysis(handle, matA, matB, alg, *dBuffer));
}

// allocate csr arrays on device memory and set them to sparse matrix descriptor
template<typename T>
void Sparse_mat<T>::setCsrArrays(cusparseSpMatDescr_t* spMatDescr,
								 int32_t rowsSize,
								 int32_t nnz, 
								 int32_t** d_csrRowsOffsets,
								 bool rowsAlloc,
								 int32_t** d_csrColInd,
								 bool colsAlloc,
								 T** d_csrValues,
							     bool valuesAlloc) {

	// allocate the other neccesary arrays - values and column indices arrays (of size nnz x 1)
	if(!rowsAlloc)checkCudaErrors(cudaMalloc((void**)d_csrRowsOffsets, (rowsSize + 1) * sizeof(int32_t))); // REMEMBER TO FREE
	if(!colsAlloc)checkCudaErrors(cudaMalloc((void**)d_csrColInd, nnz * sizeof(int32_t))); // REMEMBER TO FREE
	if(!valuesAlloc)checkCudaErrors(cudaMalloc((void**)d_csrValues, nnz * sizeof(T))); // REMEMBER TO FREE

	// set all pointers to the sparse matrix descriptor
	checkCuSparseErrors(cusparseCsrSetPointers(*spMatDescr, *d_csrRowsOffsets, *d_csrColInd, *d_csrValues));
}

// convert dense to CSR - returns three csr arrays - automatically allocated on device
template<typename T>
void Sparse_mat<T>::convertDenseToCsrFormat(cusparseHandle_t handle,
										    cusparseSpMatDescr_t* matB,
										    int32_t rows,
										    int32_t cols,
										    int32_t ld,
											const T* values,
											memLocation memLoc,
										    int32_t** d_csrColInd,
										    T** d_csrValues,
										    int32_t** d_csrRowOffsets,
											int32_t* d_nnz) {
	
    cusparseDenseToSparseAlg_t alg = CUSPARSE_DENSETOSPARSE_ALG_DEFAULT;
	// malloc csr rows offsets - size rows + 1 x 1
	checkCudaErrors(cudaMalloc((void**)d_csrRowOffsets, (rows + 1) * sizeof(int32_t))); // REMEMBER TO FREE
	// build a dense matrix from the current matrix - if the matrix is on the host, allocate memory on device
	cusparseDnMatDescr_t matA; T* d_values;
	createDenseFormat(&matA, rows, cols, ld, values, &d_values);
	// create csr formatted sparse matrix
	void* dummyCols; void* dummyVals;
	createCsrSparse(matB, rows, cols, 0, (void**)d_csrRowOffsets, &dummyCols, &dummyVals);
	// allocate external buffer if needed
	size_t bufferSize; T* dBuffer;
	alloc_analyze_convertBuffer(handle, matA, *matB, alg, &bufferSize, (void**)&dBuffer);
	// get the nnz
	int64_t rows_tmp, cols_tmp, nnz;
	checkCuSparseErrors(cusparseSpMatGetSize(*matB, &rows_tmp, &cols_tmp, &nnz));

	// allocate all csr neccesary arrays and set them
	setCsrArrays(matB, (int32_t)rows, (int32_t)nnz, d_csrRowOffsets, true, d_csrColInd, false, d_csrValues, false);
	
	// convert
	*d_nnz = (int32_t)nnz;
	checkCuSparseErrors(cusparseDenseToSparse_convert(handle, matA, *matB, alg, dBuffer));
	// free memory
	cudaFreeMem((void*)dBuffer, (void*)d_values);
	checkCuSparseErrors(cusparseDestroyDnMat(matA));
}

// get first buffer and buffer workspace estimation for sparse-sparse spGEMM matmul
template<typename T>
void Sparse_mat<T>::getWorkEstimation_allocBuffers(cusparseHandle_t handle,
												   const void* alpha,
												   cusparseSpMatDescr_t matA,
												   cusparseSpMatDescr_t matB,
												   const void* beta,
												   cusparseSpMatDescr_t matC,
												   cudaDataType computeType,
												   cusparseSpGEMMDescr_t spgemmDescr,
												   size_t* bufferSize1,
												   void** externalBuffer1) {

	// request buffer1
	checkCuSparseErrors(cusparseSpGEMM_workEstimation(handle,
													  CUSPARSE_OPERATION_NON_TRANSPOSE,
												      CUSPARSE_OPERATION_NON_TRANSPOSE,
												      alpha,
													  matA, matB,
		                                              beta,
													  matC,
													  computeType,
													  CUSPARSE_SPGEMM_DEFAULT,
													  spgemmDescr,
													  bufferSize1,
													  NULL));
	// allocate given workspace
	checkCudaErrors(cudaMalloc(externalBuffer1, *bufferSize1)); // REMEMBER TO FREE

	// allocate in descriptor
	checkCuSparseErrors(cusparseSpGEMM_workEstimation(handle,
													  CUSPARSE_OPERATION_NON_TRANSPOSE,
													  CUSPARSE_OPERATION_NON_TRANSPOSE,
													  alpha,
													  matA, matB,
													  beta,
													  matC,
													  computeType,
		                                              CUSPARSE_SPGEMM_DEFAULT,
													  spgemmDescr,
													  bufferSize1,
													  *externalBuffer1));
}

// compute the actual result and allocate second buffer for copy and computation
template<typename T>
void Sparse_mat<T>::computeSpGEMM_allocBuffers(cusparseHandle_t handle,
											   const void* alpha,
											   cusparseSpMatDescr_t matA,
											   cusparseSpMatDescr_t matB,
											   const void* beta,
											   cusparseSpMatDescr_t matC,
											   cudaDataType computeType,
											   cusparseSpGEMMDescr_t spgemmDescr,
											   size_t* bufferSize2,
											   void** externalBuffer2) {

	// get the bufferseize for work
	checkCuSparseErrors(cusparseSpGEMM_compute(handle,
											   CUSPARSE_OPERATION_NON_TRANSPOSE,
											   CUSPARSE_OPERATION_NON_TRANSPOSE,
											   alpha,
											   matA, matB,
											   beta,
											   matC,
											   computeType,
											   CUSPARSE_SPGEMM_DEFAULT,
											   spgemmDescr,
											   bufferSize2,
											   NULL));

	// allocate given workspace
	checkCudaErrors(cudaMalloc(externalBuffer2, *bufferSize2)); // REMEMBER TO FREE

	// do the actual computation
	checkCuSparseErrors(cusparseSpGEMM_compute(handle,
											   CUSPARSE_OPERATION_NON_TRANSPOSE,
											   CUSPARSE_OPERATION_NON_TRANSPOSE,
											   alpha,
											   matA, matB,
											   beta,
											   matC,
											   computeType,
											   CUSPARSE_SPGEMM_DEFAULT,
											   spgemmDescr,
											   bufferSize2,
											   *externalBuffer2));
}

// compute result with spGEMM algorithm
template<typename T>
void Sparse_mat<T>::spGEMM(cusparseHandle_t handle,
						   const void* alpha,
						   cusparseSpMatDescr_t matA,
						   cusparseSpMatDescr_t matB,
						   const void* beta,
						   cusparseSpMatDescr_t matC,
						   cudaDataType computeType,
						   cusparseSpGEMMDescr_t spgemmDescr,
						   size_t* bufferSize1,
						   void** externalBuffer1,
						   size_t* bufferSize2,
						   void** externalBuffer2) {

	getWorkEstimation_allocBuffers(handle,
								   alpha,
								   matA, matB,
								   beta,
								   matC,
								   computeType,
								   spgemmDescr,
								   bufferSize1,
								   externalBuffer1);

	computeSpGEMM_allocBuffers(handle,
							   alpha,
							   matA, matB,
							   beta,
							   matC,
							   computeType,
							   spgemmDescr,
							   bufferSize2,
							   externalBuffer2);
}

// copy the results to a seperate matrix - from descriptor
template<typename T>
void Sparse_mat<T>::copySpGEMM_toMat(cusparseHandle_t handle,
									 const void* alpha,
									 cusparseSpMatDescr_t matA,
									 cusparseSpMatDescr_t matB,
									 const void* beta,
									 cusparseSpMatDescr_t matC,
									 cudaDataType computeType,
									 cusparseSpGEMMDescr_t spgemmDescr) {

	checkCuSparseErrors(cusparseSpGEMM_copy(handle,
											CUSPARSE_OPERATION_NON_TRANSPOSE,
											CUSPARSE_OPERATION_NON_TRANSPOSE,
											alpha,
											matA, matB,
											beta,
											matC, 
											computeType,
											CUSPARSE_SPGEMM_DEFAULT,
											spgemmDescr));
}

// compute sparse-sparse matrix multiplication - C(M,N) = A(M,K)B(K,N) => C dims are MxB_cols
template<typename T>
Sparse_mat<T> Sparse_mat<T>::sparseSparseMatMul(cusparseSpMatDescr_t matB, int32_t B_cols) {
	Sparse_mat<T> C(M, (int)B_cols, memLocation::DEVICE);
	// create C handle
	void* dR; void* dC; void* dV;
	createCsrSparse(&C.SpMatDescr, (int32_t)C.M, (int32_t)C.N, 0, &dR, &dC, &dV); cusparseHandle_t handle;
	checkCuSparseErrors(cusparseCreate(&handle));

	// set parameters
	const T alpha = static_cast<T>(1);
	const T beta = static_cast<T>(0);
	cudaDataType computeType = cusparseM_T();
	size_t bufferSize1; size_t bufferSize2; void* dBuffer1; void* dBuffer2;

	// create spGEMM descriptor
	cusparseSpGEMMDescr_t spgemmDescr;
	checkCuSparseErrors(cusparseSpGEMM_createDescr(&spgemmDescr));
	
	// get workspace and compute
	spGEMM(handle,
           (const void*)&alpha,
		   SpMatDescr, matB,
		   (const void*)&beta,
		   C.SpMatDescr,
		   computeType,
		   spgemmDescr,
		   &bufferSize1,
		   &dBuffer1,
		   &bufferSize2,
		   &dBuffer2);

	// get nnz after computation
	int64_t C_rows, C_cols, C_nnz;
	checkCuSparseErrors(cusparseSpMatGetSize(C.SpMatDescr, &C_rows, &C_cols, &C_nnz));

	// update and copy result to C - need to do C.destroyCSR() after use
	C.M = (int)C_rows; C.N = (int)C_cols; C.nnz = (int32_t)C_nnz;
	
	setCsrArrays(&C.SpMatDescr, C.M, (int32_t)C_nnz, &C.csrRowOffsets, false, &C.csrColInd, false, &C.csrValues, false);

	// copy data to C csr format
	copySpGEMM_toMat(handle, &alpha, SpMatDescr, matB, &beta, C.SpMatDescr, computeType, spgemmDescr);

	// destroy handles
	cudaFreeMem((void*)dBuffer1, (void*)dBuffer2);
	checkCuSparseErrors(cusparseSpGEMM_destroyDescr(spgemmDescr));
	checkCuSparseErrors(cusparseDestroy(handle));

	return C;
}

// get CSR data from sparse matrix descriptor
template<typename T>
void Sparse_mat<T>::getCSR_data(int64_t* rows,
								int64_t* cols,
								int64_t* nnz,
								void** csrRowOffsets,
								void** csrColInd,
								void** csrValues,
								cusparseIndexType_t* csrRowOffsetsType,
								cusparseIndexType_t* csrColIndType){

	cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
	cudaDataType valueType = cusparseM_T();
	checkCuSparseErrors(cusparseCsrGet(SpMatDescr,
									   rows, cols,
									   nnz,
									   csrRowOffsets,
									   csrColInd,
									   csrValues,
									   csrRowOffsetsType,
									   &idxBase,
									   valueType));
}

// convert back from csr to dense matrix format
template<typename T>
void Sparse_mat<T>::convertCSRformatToDense(cusparseSpMatDescr_t matA, T** values, int rows, int cols, int ld) {

	// get buffer size and allocate buffer
	cusparseHandle_t handle;
	cusparseDnMatDescr_t matB;
	size_t bufferSize; void* buffer;
	// create handle
	checkCuSparseErrors(cusparseCreate(&handle));
	// create dense matrix
	T* d_values;
	checkCudaErrors(cudaMalloc((void**)&d_values, rows * cols * sizeof(T)));
	createDenseFormat(&matB, (int32_t)rows, (int32_t)cols, (int32_t)ld, d_values);

	checkCuSparseErrors(cusparseSparseToDense_bufferSize(handle,
														 matA, matB,
														 CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
														 &bufferSize));
	checkCudaErrors(cudaMalloc(&buffer, bufferSize));

	// convert
	checkCuSparseErrors(cusparseSparseToDense(handle, matA, matB, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, buffer));

	// get dense format data
	checkCudaErrors(cudaMemcpy(*values, d_values, rows * cols * sizeof(T), cudaMemcpyDeviceToHost));

	// free memory
	cudaFreeMem((void*)buffer, (void*)d_values);
	checkCuSparseErrors(cusparseDestroyDnMat(matB));
	checkCuSparseErrors(cusparseDestroy(handle));
}

// multiplication operator for structured-dense matrices
template<typename T>
Sparse_mat<T> Sparse_mat<T>::operator*(const mat<T>& B) {
	Sparse_mat<T> D(M, B.N, memLocation::DEVICE);
	// create a C matrix for output
	T* C;
	checkCudaErrors(cudaMalloc((void**)&C, M * B.N * sizeof(T)));
	checkCudaErrors(cudaMemset((void*)C, 0, M * B.N * sizeof(T)));

	// multiply D(M,B.N) = A(M,N)B(N,B.N)
	structuredDenseMatMul_op(B.M, B.N, B.data, B.memState, M, B.N, C, memLocation::DEVICE, &D.data);
	
	// free memory
	checkCudaErrors(cudaFree(C));

	return D;
}

// multiplication operator for sparse-sparse matrices
template<typename T>
Sparse_mat<T> Sparse_mat<T>::operator*(const Sparse_mat<T>& B) {
	Sparse_mat<T> C = sparseSparseMatMul(B.SpMatDescr, B.N);
	// convert data in csr format back to standard data
	convertCSRformatToDense(C.SpMatDescr, &C.data, C.M, C.N, C.M);
	return C;
}

// multiply and store the result in A - sparse-dense multiplication
template<typename T>
Sparse_mat<T>& Sparse_mat<T>::operator*=(const mat<T>& B) {
	Sparse_mat<T> D(M, B.N, memLocation::DEVICE);
	// create a C matrix for output
	T* C;
	checkCudaErrors(cudaMalloc((void**)&C, M * B.N * sizeof(T)));
	checkCudaErrors(cudaMemset((void*)C, 0, M * B.N * sizeof(T)));

	// multiply D(M,B.N) = A(M,N)B(N,B.N)
	structuredDenseMatMul_op(B.M, B.N, B.data, B.memState, M, B.N, C, memLocation::DEVICE, &D.data);

	// free memory
	checkCudaErrors(cudaFree(C));

	// copy back to this matrix
	(HOST_ALLOC(memState) && N == B.N)
		? copyDeviceToHost(D.data, data, M, N) : copyDeviceToDevice(D.data, data, M, N);

	return *this;
}

// multiply and store the result in A. for sparse-sparse matrices
template<typename T>
Sparse_mat<T>& Sparse_mat<T>::operator*=(const Sparse_mat<T>& B) {
	Sparse_mat<T> C = sparseSparseMatMul(B.SpMatDescr, B.N);

	// convert data in csr format back to standard data
	convertCSRformatToDense(C.SpMatDescr, &C.data, C.M, C.N, C.M);

	// copy the matrix to memory
	(HOST_ALLOC(memState) && N == B.N)
		? copyDeviceToHost(C.data, data, M, N) : copyDeviceToDevice(C.data, data, M, N);

	return *this;
}

/*
---------------------------------------------------------------------------------------------------
--------------------------------------------VECTOR CLASS-------------------------------------------
---------------------------------------------------------------------------------------------------
*/

// probably vectors are more dense than matrices 
// vector class supports dense vectors only.
template<typename T>
class vector : public mat<T> {
public:
	// inherit constructors
	using mat<T>::mat;
	
	// dense vector descriptor for sparse matrix - dense vector operations
	cusparseDnVecDescr_t dnVecDescr;
	bool dnDescrEnabled = false;

	// assignment operators - vector to vector
	vector<T> operator+(const vector<T>& v);
	vector<T>& operator+=(const vector<T>& v);
	// vec1(1xM)*vec2(Mx1)
	T operator*(const vector<T>& v);
	// vec1(Mx1)*vec2(1xM)
	mat<T> operator^(const vector<T>& v);

	// assignment operators - matrix to vector
	friend vector<T> operator%(const Sparse_mat<T>& M, vector<T>& v) {
		return v.sparseMatDenseVecMul(M, v);
	}
	friend vector<T> operator%(const mat<T>& M, const vector<T>& v) {
		return denseMatDenseVecMul(M, v);
	}

	// compute sparse-dense multiplication
	vector<T> sparseMatDenseVecMul(const Sparse_mat<T>& M, vector<T>& v);

	// create and destroy dense vector descriptor
	void createDense(T* values);
	void destroyDense();

private:
	void checkVectorValidity(int n, int m);

	// vector addition kernel
	void vector_addition(T* v1, T* v2, T* res, int N, memLocation v1Loc, memLocation v2Loc);
	// vector dot kernel
	void vector_dot(T* v1, T* v2, T* res, int N, memLocation v1Loc, memLocation v2Loc);
	// vector multiplication -> matrix output kernel
	void vector_mult(T* v1, T* v2, T* res, int ld1, int ld2, memLocation v1Loc, memLocation v2Loc);
	/*
	---------------- cusparse operation Y = alpha*op(A)*X + beta*Y ----------------
	*/
	// estimate buffer size needed for sparse matrix - dense vector multiplication
	void estimateBufferSize(cusparseHandle_t handle,
							const void* alpha,
							cusparseSpMatDescr_t matA,
							cusparseDnVecDescr_t vecX,
							const void* beta,
							cusparseDnVecDescr_t vecY,
							size_t* bufferSize);

	// compute spaese matrix - dense vector multiplication
	void spMv_compute(cusparseHandle_t handle,
					  const void* alpha,
					  cusparseSpMatDescr_t matA,
					  cusparseDnVecDescr_t vecX,
					  const void* beta,
					  cusparseDnVecDescr_t vecY,
					  void* externalBuffer);

	// create compute pipeline and calculate multiplication result
	void spMv(cusparseHandle_t handle,
			  cusparseSpMatDescr_t matA,
			  cusparseDnVecDescr_t vecX,
			  cusparseDnVecDescr_t vecY);
};

// implementations

// create dense vector descriptor for cusparse
template<typename T>
void vector<T>::createDense(T* values) {
	dnDescrEnabled = true;
	checkCuSparseErrors(cusparseCreateDnVec(&dnVecDescr, M, (void*)values, m_T()));
}

// destory dense cusparse descriptor
template<typename T>
void vector<T>::destroyDense() {
	dnDescrEnabled = false;
	checkCuSparseErrors(cusparseDestroyDnVec(dnVecDescr));
}

// check if the number of columns is 1. if not -> throw invalid argument
template<typename T>
void vector<T>::checkVectorValidity(int m, int n) {
	if (m <= 1 || n != 1)throw std::invalid_argument("input dimensions do not represent a vector");
}

/*
---                                                                                               ---
----------------------------- cuda c - basic vector - vector operations -----------------------------
---																				        ---
*/

// vector addition kernel
template<typename T>
__global__ void vecAdd(T* v1, T* v2, T* res, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {
		res[i] = v1[i] + v2[i];
	}
}

// vector addition function to invoke the kernel
template<typename T>
void vector<T>::vector_addition(T* v1, T* v2, T* res, int N, memLocation v1Loc, memLocation v2Loc) {
	// check if inputs are allocated in device 
	T* d_v1; T* d_v2; 
	bool v1_st = checkToAllocateOnDevice(v1, &d_v1, N, 1, v1Loc);
	bool v2_st = checkToAllocateOnDevice(v2, &d_v2, N, 1, v2Loc);

	// do the calculation
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	vecAdd <<<blocksPerGrid, threadsPerBlock>>> ((v1_st) ? d_v1 : v1,
												 (v2_st) ? d_v2 : v2,
												 res, N);

	// free memory
	if (v1_st)checkCudaErrors(cudaFree(d_v1));
	if (v2_st)checkCudaErrors(cudaFree(d_v2));
}

// addition operator
template<typename T>
vector<T> vector<T>::operator+(const vector<T>& v) {
	checkVectorValidity(v.M, v.N); checkVectorValidity(M, N);
	if (v.M != M)throw std::length_error("vector sizes do not match");

	// addition kernel
	vector<T> res(M, 1, memLocation::DEVICE);
	vector_addition(data, v.data, res.data, M, memState, v.memState);

	return res;
}

template<typename T>
vector<T>& vector<T>::operator+=(const vector<T>& v) {
	checkVectorValidity(v.M, v.N); checkVectorValidity(M, N);
	if (v.M != M)throw std::length_error("vector sizes do not match");

	// addition kernel
	vector_addition(data, v.data, data, M, memState, v.memState);
	return *this;
}

// basic dot product kernel
template<typename T>
__global__ void dotKernel(T* v1, T* v2, T* res, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {
		*res += v1[i] * v2[i];
	}
}

// dot function
template<typename T>
void vector<T>::vector_dot(T* v1, T* v2, T* res, int N, memLocation v1Loc, memLocation v2Loc) {
	// check if inputs are allocated in device 
	T* d_v1; T* d_v2; T* d_res;
	bool v1_st = checkToAllocateOnDevice(v1, &d_v1, N, 1, v1Loc);
	bool v2_st = checkToAllocateOnDevice(v2, &d_v2, N, 1, v2Loc);
	
	// assuming res is in host memory (to return to main program)
	checkCudaErrors(cudaMalloc((void**)&d_res, sizeof(T)));

	// calculate dot product
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	dotKernel<T><<<blocksPerGrid, threadsPerBlock>>> ((v1_st) ? d_v1 : v1,
													  (v2_st) ? d_v2 : v2,
													  d_res, N);

	// copy the result to host 
	checkCudaErrors(cudaMemcpy(res, d_res, sizeof(T), cudaMemcpyDeviceToHost));

	// free memory
	checkCudaErrors(cudaFree(d_res));
	if (v1_st)checkCudaErrors(cudaFree(d_v1));
	if (v2_st)checkCudaErrors(cudaFree(d_v2));
}

// dot product
template<typename T>
T vector<T>::operator*(const vector<T>& v) {
	checkVectorValidity(v.M, v.N); checkVectorValidity(M, N);
	if (v.M != M)throw std::length_error("vector sizes do not match");

	T res = static_cast<T>(0);
	vector_dot(data, v.data, &res, M, memState, v.memState);
	return res;
}

// vector multiplication kernel
template<typename T>
__global__ void vecMul(T* v1, T* v2, T* out, int ld1, int ld2) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < ld1 && y < ld2) {
		out[ld1 * y + x] = v1[x] * v2[y];
	}
}

// computes v1(mx1)*v2(1xn) = M(mxn)
template<typename T>
void vector<T>::vector_mult(T* v1, T* v2, T* res, int ld1, int ld2, memLocation v1Loc, memLocation v2Loc) {
	// check if inputs are allocated in device 
	T* d_v1; T* d_v2;
	bool v1_st = checkToAllocateOnDevice(v1, &d_v1, ld1, 1, v1Loc);
	bool v2_st = checkToAllocateOnDevice(v2, &d_v2, ld2, 1, v2Loc);

	// calculate matrix result - two dimensional blocks + threads
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
	vecMul<T> <<<numBlocks, threadsPerBlock>>> ((v1_st) ? d_v1 : v1,
												(v2_st) ? d_v2 : v2,
												res, ld1, ld2);

	// free memory
	if (v1_st)checkCudaErrors(cudaFree(d_v1));
	if (v2_st)checkCudaErrors(cudaFree(d_v2));
}

// matrix product
template<typename T>
mat<T> vector<T>::operator^(const vector<T>& v) {
	checkVectorValidity(v.M, v.N); checkVectorValidity(M, N);
	if (v.M != M)throw std::length_error("vector sizes do not match");

	// calculate v1*v2^T = M
	mat<T> resMat(M, v.M, memLocation::DEVICE);
	vector_mult(data, v.data, resMat.data, M, v.M, memState, v.memState);
	return resMat;
}


/*
---                                                                                               ---
------------------------- cublas dense matrix - dense vector multiplication -------------------------
---																				        ---
*/

// overloads for matrix-vector multiplications
void cublasGemV(cublasHandle_t handle,
				int m, int n,
				const float* alpha,
				const float* A, int lda,
				const float* x, int incx,
				const float* beta,
				float* y, int incy) {

	checkCuBLAS_status(cublasSgemv_v2(handle, CUBLAS_OP_N, m, n, alpha, A, lda, x, incx, beta, y, incy));
}

void cublasGemV(cublasHandle_t handle,
				int m, int n,
				const double* alpha,
				const double* A, int lda,
				const double* x, int incx,
				const double* beta,
				double* y, int incy) {

	checkCuBLAS_status(cublasDgemv_v2(handle, CUBLAS_OP_N, m, n, alpha, A, lda, x, incx, beta, y, incy));
}

void cublasGemV(cublasHandle_t handle,
				int m, int n,
				const cuComplex* alpha,
				const cuComplex* A, int lda,
				const cuComplex* x, int incx,
				const cuComplex* beta,
				cuComplex* y, int incy) {

	checkCuBLAS_status(cublasCgemv_v2(handle, CUBLAS_OP_N, m, n, alpha, A, lda, x, incx, beta, y, incy));
}

void cublasGemV(cublasHandle_t handle,
				int m, int n,
				const cuDoubleComplex* alpha,
				const cuDoubleComplex* A, int lda,
				const cuDoubleComplex* x, int incx,
				const cuDoubleComplex* beta,
				cuDoubleComplex* y, int incy) {

	checkCuBLAS_status(cublasZgemv_v2(handle, CUBLAS_OP_N, m, n, alpha, A, lda, x, incx, beta, y, incy));
}

// wrapper for general type gemv
template<typename T>
void GemvWrapper(cublasHandle_t handle,
				 int m, int n,
				 const T* A, int lda,
				 const T* x, T** y) {

	// set alpha, beta
	const T alpha = static_cast<T>(1);
	const T beta = static_cast<T>(0);

	cublasGemV(handle, m, n, &alpha, A, lda, x, 1, &beta, *y, 1);
}

// the following functions calculates dense matrix-dense vector multiplication
// and returns the resulting data in vector form. the returned vector is not dense descriptor initiated!
template<typename T>
vector<T> denseMatDenseVecMul(const mat<T>& M, const vector<T>& v) {
	// create vector Y
	vector<T> u(M.M, v.N, memLocation::DEVICE);

	// create handle
	cublasHandle_t handle;
	checkCuBLAS_status(cublasCreate_v2(&handle));

	// compute
	GemvWrapper(handle, M.M, M.N, M.data, M.M, v.data, &u.data);

	// free memory
	checkCuBLAS_status(cublasDestroy_v2(handle));
	return u;
}

/*
---                                                                                               ---
----------------------- cusparse sparse matrix - dense vector multiplication ------------------------
---																				        ---
*/

template<typename T>
void vector<T>::estimateBufferSize(cusparseHandle_t handle,
								   const void* alpha,
								   cusparseSpMatDescr_t matA,
								   cusparseDnVecDescr_t vecX,
								   const void* beta,
								   cusparseDnVecDescr_t vecY,
								   size_t* bufferSize) {
	
	// get buffer size needed for computation
	checkCuSparseErrors(cusparseSpMV_bufferSize(handle,
												CUSPARSE_OPERATION_NON_TRANSPOSE,
												alpha,
												matA, vecX,
												beta,
												vecY, 
												m_T(), 
												CUSPARSE_SPMV_CSR_ALG1,
												bufferSize));
}

// compute the multiplication result of a sparse matrix M and dense vector v
template<typename T>
void vector<T>::spMv_compute(cusparseHandle_t handle,
							 const void* alpha,
							 cusparseSpMatDescr_t matA,
							 cusparseDnVecDescr_t vecX,
							 const void* beta,
							 cusparseDnVecDescr_t vecY,
							 void* externalBuffer) {

	// compute with external buffer - allocated in wrapper function
	checkCuSparseErrors(cusparseSpMV(handle,
									 CUSPARSE_OPERATION_NON_TRANSPOSE,
									 alpha,
									 matA, vecX,
									 beta,
									 vecY,
									 m_T(), 
									 CUSPARSE_SPMV_CSR_ALG1,
									 externalBuffer));
}

// put all multiplication pipeline together for the full computation
template<typename T>
void vector<T>::spMv(cusparseHandle_t handle,
					 cusparseSpMatDescr_t matA,
					 cusparseDnVecDescr_t vecX,
					 cusparseDnVecDescr_t vecY) {

	// set alpha and beta
	const T alpha = static_cast<T>(1);
	const T beta = static_cast<T>(0);

	// compute
	size_t bufferSize; void* externalBuffer;
	estimateBufferSize(handle,
					   (const void*)&alpha,
					   matA, vecX,
					   (const void*)&beta,
					   vecY,
					   &bufferSize);

	checkCudaErrors(cudaMalloc(&externalBuffer, bufferSize));

	spMv_compute(handle,
				 (const void*)&alpha,
				 matA, vecX,
				 (const void*)&beta,
				 vecY,
				 externalBuffer);

	// free memory
	checkCudaErrors(cudaFree(externalBuffer));
}

// wrapper function for executing sparse matrix - dense vector multiplication. used for * operator
// assumption - matrix M is CSR ready with an existing spMatDescr
template<typename T>
vector<T> vector<T>::sparseMatDenseVecMul(const Sparse_mat<T>& M, vector<T>& v) {
	// check if vector data is in device
	T* d_vector;
	bool vec_st = checkToAllocateOnDevice(v.data, &d_vector, v.M, v.N, v.memState);
	// create descriptors
	v.createDense((vec_st) ? d_vector : v.data); // remember to free
	
	// create cusparse handle
	cusparseHandle_t handle;
	checkCuSparseErrors(cusparseCreate(&handle));

	// compute - the dense vector descriptor is destroyed due to it not being copied to the output.
	vector<T> u(M.M, v.N, memLocation::DEVICE); u.createDense(u.data); // remember to free
	spMv(handle, M.SpMatDescr, v.dnVecDescr, u.dnVecDescr); 

	// free memory
	checkCudaErrors(cudaFree(d_vector));
	checkCuSparseErrors(cusparseDestroy(handle));
	return u;
}

/*
---------------------------------------------------------------------------------------------------
---------------------------------------LINEAR EQUATION SOLVER--------------------------------------
---------------------------------------------------------------------------------------------------
*/

template<typename T>
class LinearSolver {
public:

	// types
	DECOMP _fact;
	bool _sparse;
	// for detecting whether b is not empty and recently
	// scanned an input. used for dense solver to determine
	// when to copy from b to solution. 
	bool vector_inserted = false;
	// inputs 
	mat<T> D_A;
	Sparse_mat<T> S_A;
	vector<T> b;

	// output
	vector<T> solution;

	LinearSolver(DECOMP _FACT, bool sparse) {
		_fact = _FACT;
		_sparse = sparse;
	}

	// define inputs
	void I_matrix(mat<T>& A);
	void I_matrix(Sparse_mat<T>& A);
	void I_vector(vector<T>& v);

	// define solver function
	vector<T> Solve();

private:
	// extract sparse matrix data to solve sparse linear equaitions of
	// the form Ax = b
	/*
	the general solver chooses from three different methods for solving the linear equation:
	1) LU - Ax = LUx = b => x = U^-1 * L^-1 * b,  L is lower triangular, U is upper triangular
	2) QR - Ax = QRx = b => x = Q^T * R^-1 * b, Q is orthogonal, R is upper triangular
	3) cholesky - Ax = LL^H * x = b => x = L^-1 * L^-H * b, L is a lower triangular matrix,
	   H is the hermitian operator L^H = complex_conjugate(L^T)
	*/
	void spGENERALsolver(cusolverSpHandle_t handle, const cusparseMatDescr_t descrA);

	// does the same as spGENERALsolver, for dense matrices
	/*
	1) LU - uses cusolver<t>getrf() to factorize A into LU, and cusolver<t>getrs() to solve
	2) QR - uses cusolver<t>geqrf() to factorize into QR, cusolver<t>ormqr() to transpose Q and
	   cublas<t>trsm() to solve for Rx = B, B = Q^T * b;
	3) cholesky - not available. TODO : implement batched cholesky decomposition and solver
	*/
	void dnGENERALsolver(cusolverDnHandle_t cusolverHandle);
};

// copy matrix data to new matrix
template<typename T>
void LinearSolver<T>::I_matrix(mat<T>& A) {
	if (_sparse)printf("\nsparse representation initialized - general matrix implementation is less efficient\n");
	D_A = A; D_A.empty = false;
}

// copy sparse matrix data to a new matrix
template<typename T>
void LinearSolver<T>::I_matrix(Sparse_mat<T>& A) {
	if (!_sparse)printf("\ndense representation initialized - sparse matrix implementation is less efficient\n");
	if (!A.CSR_enabled) {
		S_A = A; S_A.empty = false;
	}
	else {
		// get sizes - nnz = A.csrRowOffsets[M] - A.csrRowOffsets[0]
		int64_t rows; int64_t cols; int64_t nnz;
		checkCuSparseErrors(cusparseSpMatGetSize(A.SpMatDescr, &rows, &cols, &nnz));
		S_A.M = (int32_t)rows; S_A.N = (int32_t)cols; S_A.nnz = (int32_t)nnz;
		//! allocate memory and copy CSR data to S_A - effectively does createCSR() without converting from dense
		//! represntation to a sparse one.
		checkCudaErrors(cudaMalloc((void**)&S_A.csrRowOffsets, (rows + 1) * sizeof(int32_t)));
		checkCudaErrors(cudaMalloc((void**)&S_A.csrColInd, nnz * sizeof(int32_t)));
		checkCudaErrors(cudaMalloc((void**)&S_A.csrValues, nnz * sizeof(T)));

		// copying data	
		asyncMemcopy<int32_t>({A.csrRowOffsets, A.csrColInd},
		 					  {S_A.csrRowOffsets, S_A.csrColInd},
							  {(rows + 1) * sizeof(int32_t), nnz * sizeof(int32_t)},
		 			          {cudaMemcpyDeviceToDevice, cudaMemcpyDeviceToDevice}, 2);
		
		asyncMemcopy<T>({A.csrValues}, {S_A.csrValues}, {nnz * sizeof(T)}, {cudaMemcpyDeviceToDevice}, 1);
		// it is not needed to create a sparse matrix descriptor due to cuSOLVER's support of general cusparseMatdescr_t only
	}
}

// copy vector data into class vector
template<typename T>
void LinearSolver<T>::I_vector(vector<T>& v) {
	vector_inserted = true;
	if (!v.dnDescrEnabled) {
		b = v; b.empty = false;
	}
	else {
		b.M = v.M; b.N = v.N; b.empty = false;
		// allocate data on b
		checkCudaErrors(cudaMalloc((void**)&b.data, v.M * v.N * sizeof(T)));
		// copy data from v to b
		asyncMemcopy<T>({ v.data }, { b.data }, { v.M * v.N * sizeof(T) }, { cudaMemcpyDeviceToDevice }, 1);
		// it is not needed to create a dense vector descriptor due to cuSOLVER's support of general cusparseMatdescr_t only
	}
}

/*
----------------- solver utility functions for cuSOLVER api -----------------
*/
/*
-----sparse LAPACK api-----
*/
/*
---------------------- SOLVING WITH LU DECOMPOSITION - Ax = b => x = U^-1 * L^-1 * b ----------------------
*/
// overload (1) - float data
void sparseLUsolver(cusolverSpHandle_t handle, 
					int n,
					int nnzA,
					const cusparseMatDescr_t descrA,
					const float *csrValA,
					const int *csrRowPtrA,
					const int *csrColIndA,
					const float *b,
					float tol,
					int reorder,
					float *x,
					int *singularity){

	checkCuSolverErrors(cusolverSpScsrlsvluHost(handle,
												n, nnzA,
												descrA,
												csrValA,
												csrRowPtrA,
												csrColIndA,
												b, tol,
												reorder,
												x,
												singularity));

}

// overload (2) - double data
void sparseLUsolver(cusolverSpHandle_t handle, 
					int n,
					int nnzA,
					const cusparseMatDescr_t descrA,
					const double *csrValA,
					const int *csrRowPtrA,
					const int *csrColIndA,
					const double *b,
					float tol,
					int reorder,
					double *x,
					int *singularity){

	checkCuSolverErrors(cusolverSpDcsrlsvluHost(handle,
												n, nnzA,
												descrA,
												csrValA,
												csrRowPtrA,
												csrColIndA,
												b, tol,
												reorder,
												x,
												singularity));

}

// overload (3) - cuComplex data - [a,b], a float, b float <-> a + bi float complex
void sparseLUsolver(cusolverSpHandle_t handle, 
					int n,
					int nnzA,
					const cusparseMatDescr_t descrA,
					const cuComplex *csrValA,
					const int *csrRowPtrA,
					const int *csrColIndA,
					const cuComplex *b,
					float tol,
					int reorder,
					cuComplex *x,
					int *singularity){

	checkCuSolverErrors(cusolverSpCcsrlsvluHost(handle,
												n, nnzA,
												descrA,
												csrValA,
												csrRowPtrA,
												csrColIndA,
												b, tol,
												reorder,
												x,
												singularity));

}


// overload (4) - cuDoubleComplex data - [a,b], a double, b double <-> a + bi double complex
void sparseLUsolver(cusolverSpHandle_t handle, 
					int n,
					int nnzA,
					const cusparseMatDescr_t descrA,
					const cuDoubleComplex *csrValA,
					const int *csrRowPtrA,
					const int *csrColIndA,
					const cuDoubleComplex *b,
					float tol,
					int reorder,
					cuDoubleComplex *x,
					int *singularity){

	checkCuSolverErrors(cusolverSpZcsrlsvluHost(handle,
												n, nnzA,
												descrA,
												csrValA,
												csrRowPtrA,
												csrColIndA,
												b, tol,
												reorder,
												x,
												singularity));

}

/*
---------------------- SOLVING WITH QR DECOMPOSITION - Ax = b => x = R^-1 * Q^T * b ----------------------
*/

// overload (1) float data
void sparseQRsolver(cusolverSpHandle_t handle, 
					int n,
					int nnzA,
					const cusparseMatDescr_t descrA,
					const float *csrValA,
					const int *csrRowPtrA,
					const int *csrColIndA,
					const float *b,
					float tol,
					int reorder,
					float *x,
					int *singularity){

	checkCuSolverErrors(cusolverSpScsrlsvqr(handle,
										    n, nnzA,
										    descrA,
											csrValA,
											csrRowPtrA,
											csrColIndA,
											b, tol,
											reorder,
											x,
											singularity));
}

// overload (2) double data
void sparseQRsolver(cusolverSpHandle_t handle, 
					int n,
					int nnzA,
					const cusparseMatDescr_t descrA,
					const double *csrValA,
					const int *csrRowPtrA,
					const int *csrColIndA,
					const double *b,
					float tol,
					int reorder,
					double *x,
					int *singularity){

	checkCuSolverErrors(cusolverSpDcsrlsvqr(handle,
										    n, nnzA,
										    descrA,
											csrValA,
											csrRowPtrA,
											csrColIndA,
											b, tol,
											reorder,
											x,
											singularity));
}

// overload (3) - cuComplex data - [a,b], a float, b float <-> a + bi float complex
void sparseQRsolver(cusolverSpHandle_t handle, 
					int n,
					int nnzA,
					const cusparseMatDescr_t descrA,
					const cuComplex *csrValA,
					const int *csrRowPtrA,
					const int *csrColIndA,
					const cuComplex *b,
					float tol,
					int reorder,
					cuComplex *x,
					int *singularity){

	checkCuSolverErrors(cusolverSpCcsrlsvqr(handle,
										    n, nnzA,
										    descrA,
											csrValA,
											csrRowPtrA,
											csrColIndA,
											b, tol,
											reorder,
											x,
											singularity));
}

// overload (4) - cuDoubleComplex data - [a,b], a double, b double <-> a + bi double complex
void sparseQRsolver(cusolverSpHandle_t handle, 
					int n,
					int nnzA,
					const cusparseMatDescr_t descrA,
					const cuDoubleComplex *csrValA,
					const int *csrRowPtrA,
					const int *csrColIndA,
					const cuDoubleComplex *b,
					float tol,
					int reorder,
					cuDoubleComplex *x,
					int *singularity){

	checkCuSolverErrors(cusolverSpZcsrlsvqr(handle,
										    n, nnzA,
										    descrA,
											csrValA,
											csrRowPtrA,
											csrColIndA,
											b, tol,
											reorder,
											x,
											singularity));
}


/*
---------------------- SOLVING WITH CHOLESKY DECOMPOSITION - Ax = b => x = L^-H * L^-1 * b ----------------------
*/

// overload (1) float data
void sparseCHOLsolver(cusolverSpHandle_t handle, 
					  int n,
					  int nnzA,
					  const cusparseMatDescr_t descrA,
					  const float *csrValA,
					  const int *csrRowPtrA,
					  const int *csrColIndA,
					  const float *b,
					  float tol,
					  int reorder,
					  float *x,
					  int *singularity){

	checkCuSolverErrors(cusolverSpScsrlsvchol(handle,
										      n, nnzA,
										      descrA,
											  csrValA,
											  csrRowPtrA,
											  csrColIndA,
											  b, tol,
											  reorder,
											  x,
											  singularity));
}

// overload (2) double data
void sparseCHOLsolver(cusolverSpHandle_t handle, 
					  int n,
					  int nnzA,
					  const cusparseMatDescr_t descrA,
					  const double *csrValA,
					  const int *csrRowPtrA,
					  const int *csrColIndA,
					  const double *b,
					  float tol,
					  int reorder,
					  double *x,
					  int *singularity){

	checkCuSolverErrors(cusolverSpDcsrlsvchol(handle,
										      n, nnzA,
										      descrA,
											  csrValA,
											  csrRowPtrA,
											  csrColIndA,
											  b, tol,
											  reorder,
											  x,
											  singularity));
}

// overload (3) - cuComplex data - [a,b], a float, b float <-> a + bi float complex
void sparseCHOLsolver(cusolverSpHandle_t handle, 
					  int n,
					  int nnzA,
					  const cusparseMatDescr_t descrA,
					  const cuComplex *csrValA,
					  const int *csrRowPtrA,
					  const int *csrColIndA,
					  const cuComplex *b,
					  float tol,
					  int reorder,
					  cuComplex *x,
					  int *singularity){

	checkCuSolverErrors(cusolverSpCcsrlsvchol(handle,
										      n, nnzA,
										      descrA,
											  csrValA,
											  csrRowPtrA,
											  csrColIndA,
											  b, tol,
											  reorder,
											  x,
											  singularity));
}

// overload (4) - cuDoubleComplex data - [a,b], a double, b double <-> a + bi double complex
void sparseCHOLsolver(cusolverSpHandle_t handle, 
					  int n,
					  int nnzA,
					  const cusparseMatDescr_t descrA,
					  const cuDoubleComplex *csrValA,
					  const int *csrRowPtrA,
					  const int *csrColIndA,
					  const cuDoubleComplex *b,
					  float tol,
					  int reorder,
					  cuDoubleComplex *x,
					  int *singularity){

	checkCuSolverErrors(cusolverSpZcsrlsvchol(handle,
										      n, nnzA,
										      descrA,
											  csrValA,
											  csrRowPtrA,
											  csrColIndA,
											  b, tol,
											  reorder,
											  x,
											  singularity));
}


// equation solver using a chosen method by user - for sparse LAPACK
template<typename T>
void LinearSolver<T>::spGENERALsolver(cusolverSpHandle_t handle, const cusparseMatDescr_t descrA){
	if(S_A.M != S_A.N) printf("\nentered a NON square matrix - cuSOLVER API cannot solve the requested equation\n");
	T tolerance = static_cast<T>(1.0/(1 << 12)); int singularity;
	if(_fact == LU){
		sparseLUsolver(handle,
	 			   	   S_A.M, S_A.nnz,
	  			       descrA,
	   			       S_A.csrValues,
	    		       S_A.csrRowOffsets,
		 		       S_A.csrColInd,
		  		       b.data,
				       tolerance,
				       0, solution.data, 
				       &singularity);
	}
	else if(_fact == QR){
		sparseQRsolver(handle,
					   S_A.M, S_A.nnz,
					   descrA,
					   S_A.csrValues,
					   S_A.csrRowOffsets,
					   S_A.csrColInd,
					   b.data,
					   tolerance,
					   0, solution.data, 
					   &singularity);
	}
	else if(_fact == CHOL){
		sparseCHOLsolver(handle,
					     S_A.M, S_A.nnz,
					     descrA,
					     S_A.csrValues,
					     S_A.csrRowOffsets,
					     S_A.csrColInd,
					     b.data,
					     tolerance,
					     0, solution.data, 
					     &singularity);
	}
	
	printf("\n U matrix singularity is : %d\n", singularity); 
}

/*
-----dense LAPACK (like) api-----
*/

// check for status of cusolver routins with the devInfo variable returned by 
// cusolver routines (mostly for decomposition)
void checkDevInfo(int* d_devInfo){
	//! copying status to a host variable
	int h_devInfo;
	checkCudaErrors(cudaMemcpy(&h_devInfo, d_devInfo, sizeof(int), cudaMemcpyDeviceToHost));

	//! checking
	if(h_devInfo){
		(h_devInfo < 0) ? printf("\ncuSOLVER routine have failed. wrong parameter is %d\n", -h_devInfo):
						  printf("\ncuSOLVER routine have failed. U(%d,%d) = 0\n", h_devInfo, h_devInfo);
	}
}

/*
---------------------- SOLVING WITH LU DECOMPOSITION - Ax = b => x = U^-1 * L^-1 * b ----------------------
*/

/*
--------------- decompose A into LU ---------------
*/

// overload (1) - float
void lu_decomposition(cusolverDnHandle_t handle,
					  int m,
					  int n,
					  float *A,
					  int lda,
					  int *devIpiv,
					  int *devInfo){
	// get buffer size
	int Lwork;
	checkCuSolverErrors(cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, &Lwork));
	// allocate buffer
	float* workspace;
	checkCudaErrors(cudaMalloc((void**)&workspace, Lwork * sizeof(float)));
	// decompose
	checkCuSolverErrors(cusolverDnSgetrf(handle, m, n, A, lda, workspace, devIpiv, devInfo));
}

// overload (2) - double
void lu_decomposition(cusolverDnHandle_t handle,
					  int m,
					  int n,
					  double *A,
					  int lda,
					  int *devIpiv,
					  int *devInfo){
	// get buffer size
	int Lwork;
	checkCuSolverErrors(cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, &Lwork));
	// allocate buffer
	double* workspace;
	checkCudaErrors(cudaMalloc((void**)&workspace, Lwork * sizeof(double)));
	// decompose
	checkCuSolverErrors(cusolverDnDgetrf(handle, m, n, A, lda, workspace, devIpiv, devInfo));
}

// overload (3) - cuComplex data - [a,b], a float, b float <-> a + bi float complex
void lu_decomposition(cusolverDnHandle_t handle,
					  int m,
					  int n,
					  cuComplex *A,
					  int lda,
					  int *devIpiv,
					  int *devInfo){
	// get buffer size
	int Lwork;
	checkCuSolverErrors(cusolverDnCgetrf_bufferSize(handle, m, n, A, lda, &Lwork));
	// allocate buffer
	cuComplex* workspace;
	checkCudaErrors(cudaMalloc((void**)&workspace, Lwork * sizeof(cuComplex)));
	// decompose
	checkCuSolverErrors(cusolverDnCgetrf(handle, m, n, A, lda, workspace, devIpiv, devInfo));
}

// overload (4) - cuDoubleComplex data - [a,b], a double, b double <-> a + bi double complex
void lu_decomposition(cusolverDnHandle_t handle,
					  int m,
					  int n,
					  cuDoubleComplex *A,
					  int lda,
					  int *devIpiv,
					  int *devInfo){
	// get buffer size
	int Lwork;
	checkCuSolverErrors(cusolverDnZgetrf_bufferSize(handle, m, n, A, lda, &Lwork));
	// allocate buffer
	cuDoubleComplex* workspace;
	checkCudaErrors(cudaMalloc((void**)&workspace, Lwork * sizeof(cuDoubleComplex)));
	// decompose
	checkCuSolverErrors(cusolverDnZgetrf(handle, m, n, A, lda, workspace, devIpiv, devInfo));
}


/*
--------------- use the decomposition to solve Ax = b ---------------
*/

// overload (1) - float
void dense_lu_solver(cusolverDnHandle_t handle,
					 int n,
					 float** A,
					 int lda,
					 float* B,
					 int ldb,
					 int* devInfo){

	// decompose and check status		
	int* devIpiv; int* devInfoLU; checkCudaErrors(cudaMalloc((void**)&devInfoLU, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&devIpiv, n * sizeof(int)));
	lu_decomposition(handle, n, n, *A, lda, devIpiv, devInfoLU);
	checkDevInfo(devInfoLU);
	// solve equation
	checkCuSolverErrors(cusolverDnSgetrs(handle, CUBLAS_OP_N, n, 1, (const float*)*A, lda,
	                                     (const int*)devIpiv, B, ldb, devInfo));
	// free memory
	cudaFreeMem((void*)devIpiv, (void*)devInfoLU);
}

// overload (2) - double
void dense_lu_solver(cusolverDnHandle_t handle,
					 int n,
					 double** A,
					 int lda,
					 double* B,
					 int ldb,
					 int* devInfo){

	// decompose and check status		
	int* devIpiv; int* devInfoLU; checkCudaErrors(cudaMalloc((void**)&devInfoLU, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&devIpiv, n * sizeof(int)));
	lu_decomposition(handle, n, n, *A, lda, devIpiv, devInfoLU);
	checkDevInfo(devInfoLU);
	// solve equation
	checkCuSolverErrors(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, (const double*)*A, lda,
	                                     (const int*)devIpiv, B, ldb, devInfo));
	// free memory
	cudaFreeMem((void*)devIpiv, (void*)devInfoLU);
}

// overload (3) - cuComplex data - [a,b], a float, b float <-> a + bi float complex
void dense_lu_solver(cusolverDnHandle_t handle,
					 int n,
					 cuComplex** A,
					 int lda,
					 cuComplex* B,
					 int ldb,
					 int* devInfo){

	// decompose and check status		
	int* devIpiv; int* devInfoLU; checkCudaErrors(cudaMalloc((void**)&devInfoLU, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&devIpiv, n * sizeof(int)));
	lu_decomposition(handle, n, n, *A, lda, devIpiv, devInfoLU);
	checkDevInfo(devInfoLU);
	// solve equation
	checkCuSolverErrors(cusolverDnCgetrs(handle, CUBLAS_OP_N, n, 1, (const cuComplex*)*A, lda,
	                                     (const int*)devIpiv, B, ldb, devInfo));
	// free memory
	cudaFreeMem((void*)devIpiv, (void*)devInfoLU);
}


void dense_lu_solver(cusolverDnHandle_t handle,
					 int n,
					 cuDoubleComplex** A,
					 int lda,
					 cuDoubleComplex* B,
					 int ldb,
					 int* devInfo){

	// decompose and check status		
	int* devIpiv; int* devInfoLU; checkCudaErrors(cudaMalloc((void**)&devInfoLU, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&devIpiv, n * sizeof(int)));
	lu_decomposition(handle, n, n, *A, lda, devIpiv, devInfoLU);
    checkDevInfo(devInfoLU);
	// solve equation
	checkCuSolverErrors(cusolverDnZgetrs(handle, CUBLAS_OP_N, n, 1, (const cuDoubleComplex*)*A, lda,
	                                     (const int*)devIpiv, B, ldb, devInfo));
	// free memory
	cudaFreeMem((void*)devIpiv, (void*)devInfoLU);
}

/*
---------------------- SOLVING WITH QR DECOMPOSITION - Ax = b => x = R^-1 * Q^T * b ----------------------
*/

//! procedure : decompose A to A = QR, QQ^T = I, R upper triangular with cusolver<t>geqrf()
//! calculate Q^T * b with cusolver<t>ormqr()
//! solve Rx = B (= Q^T * b) with cublas<t>trsm()

// overload (1) - float
void dense_qr_solver(cusolverDnHandle_t cusHandle,
					 cublasHandle_t cubHandle, 
					 int mA, int nA,
					 float* A,
					 int lda,
					 int mB, int nB,
					 float* b,
					 int ldb) {

	//! part (1) - decompose
	// get buffer sizes - for geqrf (decomposition) and ormqr(Q transpose)
	float* workspace; int Lwork_geqrf; int Lwork_ormqr; float* tau; int* devInfogeqrf;
	// allocate tau helper array
	checkCudaErrors(cudaMalloc((void**)&tau, mB * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&devInfogeqrf, sizeof(int)));
	// allocate buffers for both operations
	checkCuSolverErrors(cusolverDnSgeqrf_bufferSize(cusHandle, mA, nA, A, lda, &Lwork_geqrf));
	checkCuSolverErrors(cusolverDnSormqr_bufferSize(cusHandle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
													mB, nB, mB, A, lda, tau, b, ldb, &Lwork_ormqr));
	// calculate lwork and allocate workspace
	int Lwork = std::max(Lwork_geqrf, Lwork_ormqr);
	checkCudaErrors(cudaMalloc((void**)&workspace, Lwork * sizeof(float)));
	// decompose with geqrf
	checkCuSolverErrors(cusolverDnSgeqrf(cusHandle, mA, nA, A, lda, tau, workspace, Lwork, devInfogeqrf));

	//! part (2) - transpose
	int* devInfo_ormqr;
	checkCudaErrors(cudaMalloc((void**)&devInfo_ormqr, sizeof(int)));
	checkCuSolverErrors(cusolverDnSormqr(cusHandle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
										 mB, nB, mB, A, lda, tau, b, ldb, workspace, Lwork, devInfo_ormqr));

	// check status
	checkDevInfo(devInfogeqrf); checkDevInfo(devInfo_ormqr);
	//! part (3) - solve - b is also the result x
	const float alpha = 1.0f;
	checkCuBLAS_status(cublasStrsm_v2(cubHandle,
									  CUBLAS_SIDE_LEFT,
									  CUBLAS_FILL_MODE_UPPER,
									  CUBLAS_OP_N,
									  CUBLAS_DIAG_NON_UNIT,
									  mB, nB, &alpha, A, lda, b, ldb));

	// free memory
	cudaFreeMem((void*)tau, (void*)workspace, (void*)devInfogeqrf, (void*)devInfo_ormqr);
}

// overload (2) - double
void dense_qr_solver(cusolverDnHandle_t cusHandle,
					 cublasHandle_t cubHandle, 
					 int mA, int nA,
					 double* A,
					 int lda,
					 int mB, int nB,
					 double* b,
					 int ldb) {

	//! part (1) - decompose
	// get buffer sizes - for geqrf (decomposition) and ormqr(Q transpose)
	double* workspace; int Lwork_geqrf; int Lwork_ormqr; double* tau; int* devInfogeqrf;
	// allocate tau helper array
	checkCudaErrors(cudaMalloc((void**)&tau, mB * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&devInfogeqrf, sizeof(int)));
	// allocate buffers for both operations
	checkCuSolverErrors(cusolverDnDgeqrf_bufferSize(cusHandle, mA, nA, A, lda, &Lwork_geqrf));
	checkCuSolverErrors(cusolverDnDormqr_bufferSize(cusHandle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
													mB, nB, mB, A, lda, tau, b, ldb, &Lwork_ormqr));
	// calculate lwork and allocate workspace
	int Lwork = std::max(Lwork_geqrf, Lwork_ormqr);
	checkCudaErrors(cudaMalloc((void**)&workspace, Lwork * sizeof(double)));
	// decompose with geqrf
	checkCuSolverErrors(cusolverDnDgeqrf(cusHandle, mA, nA, A, lda, tau, workspace, Lwork, devInfogeqrf));

	//! part (2) - transpose
	int* devInfo_ormqr;
	checkCudaErrors(cudaMalloc((void**)&devInfo_ormqr, sizeof(int)));
	checkCuSolverErrors(cusolverDnDormqr(cusHandle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
										 mB, nB, mB, A, lda, tau, b, ldb, workspace, Lwork, devInfo_ormqr));

	// check status
	checkDevInfo(devInfogeqrf); checkDevInfo(devInfo_ormqr);
	//! part (3) - solve - b is also the result x
	const double alpha = 1.0f;
	checkCuBLAS_status(cublasDtrsm_v2(cubHandle,
									  CUBLAS_SIDE_LEFT,
									  CUBLAS_FILL_MODE_UPPER,
									  CUBLAS_OP_N,
									  CUBLAS_DIAG_NON_UNIT,
									  mB, nB, &alpha, A, lda, b, ldb));

	// free memory
	cudaFreeMem((void*)tau, (void*)workspace, (void*)devInfogeqrf, (void*)devInfo_ormqr);
}

// overload (3) - cuComplex data - [a,b], a float, b float <-> a + bi float complex
void dense_qr_solver(cusolverDnHandle_t cusHandle,
					 cublasHandle_t cubHandle, 
					 int mA, int nA,
					 cuComplex* A,
					 int lda,
					 int mB, int nB,
					 cuComplex* b,
					 int ldb) {

	//! part (1) - decompose
	// get buffer sizes - for geqrf (decomposition) and ormqr(Q transpose)
	cuComplex* workspace; int Lwork_geqrf; int Lwork_ormqr; cuComplex* tau; int* devInfogeqrf;
	// allocate tau helper array
	checkCudaErrors(cudaMalloc((void**)&tau, mB * sizeof(cuComplex)));
	checkCudaErrors(cudaMalloc((void**)&devInfogeqrf, sizeof(int)));
	// allocate buffers for both operations
	checkCuSolverErrors(cusolverDnCgeqrf_bufferSize(cusHandle, mA, nA, A, lda, &Lwork_geqrf));
	checkCuSolverErrors(cusolverDnCunmqr_bufferSize(cusHandle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
													mB, nB, mB, A, lda, tau, b, ldb, &Lwork_ormqr));
	// calculate lwork and allocate workspace
	int Lwork = std::max(Lwork_geqrf, Lwork_ormqr);
	checkCudaErrors(cudaMalloc((void**)&workspace, Lwork * sizeof(cuComplex)));
	// decompose with geqrf
	checkCuSolverErrors(cusolverDnCgeqrf(cusHandle, mA, nA, A, lda, tau, workspace, Lwork, devInfogeqrf));

	//! part (2) - transpose
	int* devInfo_ormqr;
	checkCudaErrors(cudaMalloc((void**)&devInfo_ormqr, sizeof(int)));
	checkCuSolverErrors(cusolverDnCunmqr(cusHandle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
										 mB, nB, mB, A, lda, tau, b, ldb, workspace, Lwork, devInfo_ormqr));

	// check status
	checkDevInfo(devInfogeqrf); checkDevInfo(devInfo_ormqr);
	//! part (3) - solve - b is also the result x
	const cuComplex alpha = static_cast<cuComplex>(make_cuComplex(1.0f, 0.0f));
	checkCuBLAS_status(cublasCtrsm_v2(cubHandle,
									  CUBLAS_SIDE_LEFT,
									  CUBLAS_FILL_MODE_UPPER,
									  CUBLAS_OP_N,
									  CUBLAS_DIAG_NON_UNIT,
									  mB, nB, &alpha, A, lda, b, ldb));

	// free memory
	cudaFreeMem((void*)tau, (void*)workspace, (void*)devInfogeqrf, (void*)devInfo_ormqr);
}

// overload (4) - cuDoubleComplex data - [a,b], a double, b double <-> a + bi double complex
void dense_qr_solver(cusolverDnHandle_t cusHandle,
					 cublasHandle_t cubHandle, 
					 int mA, int nA,
					 cuDoubleComplex* A,
					 int lda,
					 int mB, int nB,
					 cuDoubleComplex* b,
					 int ldb) {

	//! part (1) - decompose
	// get buffer sizes - for geqrf (decomposition) and ormqr(Q transpose)
	cuDoubleComplex* workspace; int Lwork_geqrf; int Lwork_ormqr; cuDoubleComplex* tau; int* devInfogeqrf;
	// allocate tau helper array
	checkCudaErrors(cudaMalloc((void**)&tau, mB * sizeof(cuDoubleComplex)));
	checkCudaErrors(cudaMalloc((void**)&devInfogeqrf, sizeof(int)));
	// allocate buffers for both operations
	checkCuSolverErrors(cusolverDnZgeqrf_bufferSize(cusHandle, mA, nA, A, lda, &Lwork_geqrf));
	checkCuSolverErrors(cusolverDnZunmqr_bufferSize(cusHandle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
													mB, nB, mB, A, lda, tau, b, ldb, &Lwork_ormqr));
	// calculate lwork and allocate workspace
	int Lwork = std::max(Lwork_geqrf, Lwork_ormqr);
	checkCudaErrors(cudaMalloc((void**)&workspace, Lwork * sizeof(cuDoubleComplex)));
	// decompose with geqrf
	checkCuSolverErrors(cusolverDnZgeqrf(cusHandle, mA, nA, A, lda, tau, workspace, Lwork, devInfogeqrf));

	//! part (2) - transpose
	int* devInfo_ormqr;
	checkCudaErrors(cudaMalloc((void**)&devInfo_ormqr, sizeof(int)));
	checkCuSolverErrors(cusolverDnZunmqr(cusHandle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
										 mB, nB, mB, A, lda, tau, b, ldb, workspace, Lwork, devInfo_ormqr));

	// check status
	checkDevInfo(devInfogeqrf); checkDevInfo(devInfo_ormqr);
	//! part (3) - solve - b is also the result x
	const cuDoubleComplex alpha = static_cast<cuDoubleComplex>(make_cuDoubleComplex(1.0f, 0.0f));
	checkCuBLAS_status(cublasZtrsm_v2(cubHandle,
									  CUBLAS_SIDE_LEFT,
									  CUBLAS_FILL_MODE_UPPER,
									  CUBLAS_OP_N,
									  CUBLAS_DIAG_NON_UNIT,
									  mB, nB, &alpha, A, lda, b, ldb));

	// free memory
	cudaFreeMem((void*)tau, (void*)workspace, (void*)devInfogeqrf, (void*)devInfo_ormqr);
}

/*
dense matrix linear equation solver
methods implemented into interface : LU decomposition, QR decomposition. CHOLESKY is available
only with sparse interface.
*/
template<typename T>
void LinearSolver<T>::dnGENERALsolver(cusolverDnHandle_t cusolverHandle){
	if(D_A.M != D_A.N) printf("\nentered a NON square matrix - cuSOLVER API cannot solve LU method request\n");
	// allocate devInfo pointer as a status for the operation
	int* devInfo; checkCudaErrors(cudaMalloc((void**)&devInfo, sizeof(int)));
	//! if a new vector is scanned - copy the vector into the solution vector. 
	//! PURPOSE: in cuSOLVER dense API the b vector is also changed and retured as the solution,
	//! the following flag is used for keeping b content and change only the solution vector.
	if(vector_inserted) solution = b;
	if(_fact == LU){
		dense_lu_solver(cusolverHandle, D_A.M, &D_A.data, D_A.M, solution.data, solution.M, devInfo);
		checkDevInfo(devInfo);
	}
	else if(_fact == QR){
		cublasHandle_t cublasHandle;
		checkCuBLAS_status(cublasCreate_v2(&cublasHandle));
		dense_qr_solver(cusolverHandle, cublasHandle, D_A.M, D_A.N, D_A.data, D_A.M,
						solution.M, solution.N, solution.data, solution.M);
		checkCuBLAS_status(cublasDestroy_v2(cublasHandle));
	}
}

// general solver - takes either dense or sparse matrices according to _sparse input.
template<typename T>
vector<T> LinearSolver<T>::Solve(){
	// allocate memory for equation solution vector
	int32_t sol_size = (_sparse) ? S_A.N : D_A.N;
	solution.M = sol_size; solution.N = 1;
	if (solution.empty)checkCudaErrors(cudaMalloc((void**)&solution.data, sol_size * sizeof(T))); // REMEMBER TO FREE
	solution.empty = false;
	// create matrix handle - default - CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO
	cusparseMatDescr_t generalMatDescr;
	checkCuSparseErrors(cusparseCreateMatDescr(&generalMatDescr));

	// solve
	if(_sparse){
		// create handle
		cusolverSpHandle_t spHandle;
		checkCuSolverErrors(cusolverSpCreate(&spHandle));
		// solve
		spGENERALsolver(spHandle, (const cusparseMatDescr_t)generalMatDescr);
		checkCuSolverErrors(cusolverSpDestroy(spHandle));
	}
	else{
		// create handle
		cusolverDnHandle_t dnHandle;
		checkCuSolverErrors(cusolverDnCreate(&dnHandle));
		// solve
		dnGENERALsolver(dnHandle);
		checkCuSolverErrors(cusolverDnDestroy(dnHandle));
	}
	//! free matrix descriptor memory - vector_inserted is set to false when the solver 
	//! works out a solution. if no new vector is scanned then b is "old" and not changed
	//! for reusing the solver. otherwise, the flag turnes true.
	checkCuSparseErrors(cusparseDestroyMatDescr(generalMatDescr)); vector_inserted = false;
	return solution;
}

