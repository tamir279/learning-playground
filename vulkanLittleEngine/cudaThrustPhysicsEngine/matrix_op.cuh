#include <cstring>
#include <cstdio>
#include <typeinfo>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void printMeDev(float* P, int uWP, int uHP) {
	//printf("\n %f",P[1]);
	int i, j;
	for (i = 0; i < uHP; i++) {
		printf("\n");
		for (j = 0; j < uWP; j++)
			printf("%f ", P[i * uHP + j]);
		//printf("%f ",P[i*uWP+j]);
	}
	printf("\n");
}

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


// three memory maps
enum class memLocation { HOST, HOST_PINNED, DEVICE };

// matrix class in column major order
template<typename T>
class mat {
public:
	// thee matrix data
	T* m;
	// dimensions
	int M; int N; memLocation memState;

	mat(int n_rows, int n_cols, memLocation memLoc) {
		// inputs
		this->M = n_rows;
		this->N = n_cols;
		this->memState = memLoc;

		// allocate memory
		if (memState == memLocation::HOST || memState == memLocation::HOST_PINNED) {
			allocateOnHost(&m, M, N, memState);
		}
		else {
			allocateOnDevice(&m, M, N);
		}
	}

	~mat() {
		(memState == memLocation::HOST || memState == memLocation::HOST_PINNED)
			? destroyHostMatrix(&m, memState) : destroyDeviceMatrix(&m);
	}

	// copy constructor
	//mat(const mat<T>& B) {
	//}

	// operators
	void operator+=(const mat<T> B) {
		// allocate result matrix on device for cublas computation
		mat<T> C(M, N, memLocation::DEVICE);

		matAdd(CUBLAS_OP_N,
			   CUBLAS_OP_N,
			   M, N,
			   (const T**)&m, M,
			   memState,
			   (const T**)&(B.m), M,
		       B.memState,
			   &(C.m), M,
			   C.memState);

		// copy back to this matrix
		(memState == memLocation::HOST || memState == memLocation::HOST_PINNED)
			? copyDeviceToHost(C.m, m, M, N) : copyDeviceToDevice(C.m, m, M, N);
	}

	// the + operator returns a device allocated matrix
	mat<T> operator+(const mat<T> B) {
		// allocate result matrix on device for cublas computation
		mat<T> C(M, N, memLocation::DEVICE);

		matAdd(CUBLAS_OP_N,
			   CUBLAS_OP_N,
			   M, N,
			   (const T**)&m, M,
			   memState,
			   (const T**)&(B.m), M,
			   B.memState,
			   &(C.m), M,
			   C.memState);

		// return the new matrix
		return C;
	}

	void operator*=(const mat<T> B) {
		// allocate result matrix on device for cublas computation
		mat<T> C(M, B.N, memLocation::DEVICE);

		//get the cuda types
		cudaDataType mT = m_T();

		// C(M,N) = A(M,N)B(N,B.N)
		matMul(CUBLAS_OP_N,
			   CUBLAS_OP_N,
			   M, B.N, N,
			   (const T**)&m, mT,
			   memState,
			   M,
			   (const T**)&(B.m), mT,
			   B.memState,
			   N,
			   &(C.m), mT,
			   M, mT,
			   CUBLAS_GEMM_DEFAULT);

		// copy back to this matrix
		(memState == memLocation::HOST || memState == memLocation::HOST_PINNED)
			? copyDeviceToHost(C.m, m, M, N) : copyDeviceToDevice(C.m, m, M, N);
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
			   (const T**)&m, mT,
			   memState,
			   M,
			   (const T**)&(B.m), mT,
			   B.memState,
			   N,
			   &(C.m), mT,
			   M, mT,
			   CUBLAS_GEMM_DEFAULT);
		printf("\ncopy multiplication matrix result:\n");
		printMeDev <<<1, 1>>> (C.m, M, B.N);
		// return the new matrix
		return C;
	}

	// copy host to device - from B to matrix m
	void operator<<=(const mat<T> B) {
		copyHostToDevice(B.m, m, M, N);
	}

	// copy device to host - from B to matrix m
	void operator>>=(const mat<T> B) {
		copyDeviceToHost(B.m, m, M, N);
	}

	// copy host to host - from B to matrix m
	void operator<=(const mat<T> B) {
		copyHostToHost(B.m, m, M, N);
	}

	// copy device to device - from B to matrix m
	void operator>=(const mat<T> B) {
		copyDeviceToDevice(B.m, m, M, N);
	}

	// dynamic copying
	void operator=(const mat<T>& B) {
		if ((memState == memLocation::HOST_PINNED || memState == memLocation::HOST) &&
			(B.memState == memLocation::HOST_PINNED || B.memState == memLocation::HOST))
			copyHostToHost(B.m, m, M, N);
		else if ((memState == memLocation::HOST_PINNED || memState == memLocation::HOST) &&
			B.memState == memLocation::DEVICE)
			copyDeviceToHost(B.m, m, M, N);
		else if (memState == memLocation::DEVICE &&
			(B.memState == memLocation::HOST_PINNED || B.memState == memLocation::HOST))
			copyHostToDevice(B.m, m, M, N);
		else if (memState == memLocation::DEVICE && B.memState == memLocation::DEVICE)
			copyDeviceToDevice(B.m, m, M, N);
		else printf_s("invalid memory location format ! \n");
	}

private:

	// decide compute and data types for matrix m - m_T matrix data type, c_T - compute type
	cudaDataType m_T();

	// memory and device management ( + overloads)
	// ver. 1 - no const

	// allocation
	void allocateOnHost(T** h_m, int n_rows, int n_cols, memLocation memLoc);
	void allocateOnDevice(T** d_m, int n_rows, int n_cols);

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

	// check if transfer to device is needed
	bool checkToAllocateOnDevice(const T* h_m, T** d_m, int n_rows, int n_cols, memLocation memLoc);

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
