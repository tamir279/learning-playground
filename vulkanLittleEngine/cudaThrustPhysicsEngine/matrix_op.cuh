#include <cstring>
#include <cstdio>
#include <typeinfo>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cusparseLt.h>

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

#define HOST_ALLOC(memLoc)((memLoc) == memLocation::HOST_PINNED || (memLoc) == memLocation::HOST)
#define DEVICE_ALLOC(memLoc)((memLoc) == memLocation::DEVICE)


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

/*
---------------------------------------------------------------------------------------------------
-------------------------------------------MATRIX CLASS--------------------------------------------
---------------------------------------------------------------------------------------------------
*/

// three memory maps
enum class memLocation { HOST, HOST_PINNED, DEVICE };

// matrix class in column major order
template<typename T>
class mat {
public:
	// thee matrix data
	T* data;
	// dimensions
	int M; int N; memLocation memState;

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

	// 
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
		(HOST_ALLOC(memState))
			? destroyHostMatrix(&data, memState) : destroyDeviceMatrix(&data);
	}

	// operators
	mat<T>& operator+=(const mat<T>& B) {
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
		copyHostToDevice(B.data, data, M, N);
		return *this;
	}

	// copy device to host - from B to matrix m
	mat<T>& operator>>=(const mat<T>& B) {
		copyDeviceToHost(B.data, data, M, N);;
		return *this;
	}

	// copy host to host - from B to matrix m
	mat<T>& operator<=(const mat<T>& B) {
		copyHostToHost(B.data, data, M, N);
		return *this;
	}

	// copy device to device - from B to matrix m
	mat<T>& operator>=(const mat<T>& B) {
		copyDeviceToDevice(B.data, data, M, N);
		return *this;
	}

	// dynamic copying
	mat<T>& operator=(const mat<T>& B) {
		dynamicCopy(data, B.data, M, N, memState, B.memState);
		return *this;
	}

	// check if transfer to device is needed
	bool checkToAllocateOnDevice(const T* h_m, T** d_m, int n_rows, int n_cols, memLocation memLoc);

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
	int64_t* csrRowOffsets; // size (M + 1) x 1 - encoding of row indices of nz elements
	int64_t* csrColInd; // size nnz x 1 - column indices of nz elements
	T* csrValues; // size nnz x 1 - values of all nz elements

	// needed for creating csr formatting and for sparse - sparse operations
	cusparseSpMatDescr_t SpMatDescr;

	// handling CSR format for sparse - sparse operations in cuSPARSE
	// includes creating a sparseMatrix handle (in createCSR) and destroying it (in destroyCSR)
	void createCSR();
	void destroyCSR();

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
						   const cusparseLtMatmulPlan_t* plan,
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
						   int64_t rows,
						   int64_t cols,
						   int64_t ld,
						   T* values);

	// create the CSR format in the sparse matrix descriptor
	void createCsrSparse(cusparseSpMatDescr_t* spMatDescr,
						 int64_t rows,
						 int64_t cols,
						 int64_t nnz,
						 void** csrRowOffsets,
						 void** csrColInd,
						 void** csrValues);

	// convert the dense format of a matrix to a sparse CSR format, allocate CSR arrays in device
	// create a dense matrix and convert it to a sparse CSR representation
	void convertDenseToCsrFormat(cusparseHandle_t handle,
								 cusparseSpMatDescr_t matB,
								 int64_t rows,
								 int64_t cols,
								 int64_t ld,
								 const T* values,
								 memLocation memLoc,
								 int64_t** d_csrColInd,
								 T** d_csrValues,
		                         int64_t** d_csrRowOffsets);

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
	Sparse_mat<T> sparseSparseMatMul(cusparseSpMatDescr_t matB, int64_t B_cols);

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
	void convertCSRformatToDense(cusparseSpMatDescr_t matA, void** values);
};

// create CSR formatted matrix 
// used when a matrix has known dimensions and values, for sparse-sparse cuSPARSE api operations only
template<typename T>
void Sparse_mat<T>::createCSR() {
	// create handle
	cusparseHandle_t handle;
	checkCuSparseErrors(cusparseCreate(&handle));
	// allocate memory for the three arrays and convert from a dense representation
	// to a CSR sparse matrix representation
	convertDenseToCsrFormat(handle,
							SpMatDescr,
							(int64_t)M,
							(int64_t)N,
							(int64_t)M,
							(const T*)data,
							memState,
							&csrColInd,
							&csrValues,
							&csrRowOffsets);
	// destroy handle
	checkCuSparseErrors(cusparseDestroy(&handle));
}

// destroy CSR format sparse matrix handle and free all csr arrays
template<typename T>
void Sparse_mat<T>::destroyCSR() {
	checkCudaErrors(cudaFree(csrRowOffsets));
	checkCudaErrors(cudaFree(csrColInd));
	checkCudaErrors(cudaFree(csrValues));
	checkCuSparseErrors(cusparseDestroySpMat(SpMatDescr));
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
	// init B, C
	cusparseLtDenseMatInit(*handle, &matDescrs[1], b_rows, b_cols, b_rows, 16, valueType);
	cusparseLtDenseMatInit(*handle, &matDescrs[2], c_rows, c_cols, c_rows, 16, valueType);

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
	checkCudaErrors(cudaMemcpyAsync(&isValid, d_valid, cudaMemcpyDeviceToHost, stream));
	checkCudaErrors(cudaStreamSynchronize(stream));
	// free d_valid
	checkCudaErrors(cudaFree(d_valid));

	// check for prune success
	if (isValid) {
		printf_s("!!!!The matrix has been pruned in a wrong way.cusparseLtMatmul will not provide correct results\n");
		return EXIT_FAILURE;
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
	checkCudaErrors(cudaMalloc((void**)&d_out, (size_t)sizeA));

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
								      const cusparseLtMatmulPlan_t* plan,
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
	bool A_status = checkToAllocateOnDevice(A, &d_A, sizeA, 1, memLocation Aloc);
	bool B_status = checkToAllocateOnDevice(B, &d_B, sizeB, 1, memLocation Bloc);
	bool C_status = checkToAllocateOnDevice(C, &d_C, sizeC, 1, memLocation Cloc);

	// search for optimal algorithm
	cusparseLtSearchOptimalKernel(handle, plan,
								  &alpha,
								  (A_status) ? d_A : A,
								  (B_status) ? d_B : B,
								  &beta,
								  (C_status) ? d_C : C,
								  *d_D,
								  workspace,
								  streams,
								  numStreams);

	// perform matrix multiplication
	checkCuSparseErrors(cusparseLtMatmul(handle, plan,
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
	// initialize all descriptors
	cusparseLtInitDescriptors(&pHandle, matDescrs, M, N, b_rows, b_cols, M, b_cols, valueType, &pMatmulDescr, &pAlgSel, &pPlan);

	// create stream
	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreate(&stream));

	// prune and compress matrix A (the sparse matrix)
	T* d_A_cmprs;
	cusparseLtSparseMatPrune_Compress(pHandle, pMatmulDescr, pPlan, (const T*)data, M * N * sizeof(T), memState, &d_A_cmprs, stream);

	// search for optimal algorithm and execute operation

	structuredDenseMatMul(pHandle,
						  pPlan,
						  M * N * sizeof(T), 
						  b_rows * b_cols * sizeof(T),
						  M * b_cols * sizeof(T),
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
	checkCuSparseErrors(cusparseDestroy(handle));
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
									  int64_t rows,
									  int64_t cols,
									  int64_t ld,
									  T* values) {

	// check if the matrix is in device or not
	T* d_values;
	auto status = checkToAllocateOnDevice((const T*)values, &d_values, rows, cols, memState);
	
	// create dense matrix handle
	cudaDataType valueType = cusparseM_T();
	checkCuSparseErrors(cusparseCreateDnMat(dnMatDescr,
											rows, cols,
											ld,
											status ? (void*)d_values : (void*)values,
											valueType,
											CUSPARSE_ORDER_COL));
	
	// free d_values
	checkCudaErrors(cudaFree(d_values));
}

// create a csr formatted sparse matrix representation
template<typename T>
void Sparse_mat<T>::createCsrSparse(cusparseSpMatDescr_t* spMatDescr,
								    int64_t rows,
								    int64_t cols,
								    int64_t nnz,
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
template<typename T>
void alloc_analyze_convertBuffer(cusparseHandle_t handle,
								 cusparseSpMatDescr_t matB,
								 cusparseDenseToSparseAlg_t alg,
								 size_t bufferSize,
								 void** dBuffer) {

	checkCuSparseErrors(cusparseDenseToSparse_bufferSize(handle, matA, matB, alg, &bufferSize));
	checkCudaErrors(cudaMalloc(dBuffer, bufferSize)); // REMEMBER TO FREE

	// update nnz in the descriptor of matB
	checkCuSparseErrors(cusparseDenseToSparse_analysis(handle, matA, matB, alg, *dBuffer));
}

// convert dense to CSR - returns three csr arrays - automatically allocated on device
template<typename T>
void Sparse_mat<T>::convertDenseToCsrFormat(cusparseHandle_t handle,
										    cusparseSpMatDescr_t matB,
										    int64_t rows,
										    int64_t cols,
										    int64_t ld,
											const T* values,
											memLocation memLoc,
										    int64_t** d_csrColInd,
										    T** d_csrValues,
										    int64_t** d_csrRowOffsets) {
	
    cusparseDenseToSparseAlg_t alg = CUSPARSE_DENSETOSPARSE_ALG_DEFAULT;
	// malloc the rows offset in device
	checkCudaErrors(cudaMalloc((void**)d_csrRowOffsets, (rows + 1) * sizeof(int64_t))); // REMEMBER TO FREE

	// build a dense matrix from the current matrix - if the matrix is on the host, allocate memory on device
	cusparseDnMatDescr_t matA;
	createDenseFormat(&matA, rows, cols, ld, const_cast<T*>(values));

	// create csr formatted sparse matrix
	createCsrSparse(matB, rows, cols, 0, d_csrRowOffsets, NULL, NULL);

	// allocate external buffer if needed
	size_t bufferSize; void* dBuffer;
	alloc_analyze_convertBuffer(handle, matA, matB, alg, bufferSize, &dBuffer);

	// get the nnz
	int64_t rows_tmp, cols_tmp, nnz;
	checkCuSparseErrors(cusparseSpMatGetSize(matB, &rows_tmp, &cols_tmp, &nnz));

	// allocate the other neccesary arrays - values and column indices arrays (of size nnz x 1)
	checkCudaErrors(cudaMalloc((void**)d_csrColInd, nnz * sizeof(int64_t))); // REMEMBER TO FREE
	checkCudaErrors(cudaMalloc((void**)d_csrValues, nnz * sizeof(T))); // REMEMBER TO FREE

	// set all pointers
	checkCuSparseErrors(cusparseCsrSetPointers(matB, d_csrRowOffsets, d_csrColInd, d_csrValues));

	// convert
	checkCuSparseErrors(cusparseDenseToSparse_convert(handle, matA, matB, alg, dBuffer));

	// free memory
	checkCudaErrors(cudaFree(dBuffer));
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
Sparse_mat<T> Sparse_mat<T>::sparseSparseMatMul(cusparseSpMatDescr_t matB, int64_t B_cols) {
	Sparse_mat<T> C(M, B_cols, memLocation::DEVICE);
	// create C handle
	createCsrSparse(&C.SpMatDescr, C.M, C.N, 0, NULL, NULL, NULL); cusparseHandle_t handle;
	checkCuSparseErrors(cusparseCreate(&handle));

	// set parameters
	const T alpha = static_cast<T>(1);
	const T beta = static_cast<T>(0);
	cudaDataType computeType = cusparseM_T();
	size_t bufferSize1, bufferSize2; void* dBuffer1, void* dBuffer2;

	// create spGEMM descriptor
	cusparseSpGEMMDescr_t spgemmDesc;
	checkCuSparseErrors(cusparseSpGEMM_createDescr(&spgemmDescr));
	
	// get workspace and compute
	spGEMM(handle,
           (const void*)&alpha,
		   SpMatDescr, matB,
		   (const void*)&beta,
		   C.SpMatDescr,
		   computeType,
		   spgemmDesc,
		   &bufferSize1,
		   &dBuffer1,
		   &bufferSize2,
		   &dBuffer2);

	// get nnz after computation
	int64_t C_rows, C_cols, C_nnz;
	checkCuSparseErrors(cusparseSpMatGetSize(C.SpMatDescr, &C_rows, &C_cols, &C_nnz));

	// update and copy result to C - need to do C.destroyCSR() after use
	C.M = C_rows; C.N = C_cols;
	checkCudaErrors(cudaMalloc((void**)&C.csrRowOffsets, (C.M + 1) * sizeof(int64_t))); // REMEMBER TO FREE
	checkCudaErrors(cudaMalloc((void**)&C.csrColInd, C_nnz * sizeof(int64_t)));  // REMEMBER TO FREE
	checkCudaErrors(cudaMalloc((void**)&C.csrValues, C_nnz * sizeof(T))); // REMEMBER TO FREE

	// copy data to C csr format
	copySpGEMM_toMat(handle, &alpha, SpMatDescr, matB, &beta, C.SpMatDescr, computeType, spgemmDesc);

	// destroy handles
	checkCudaErrors(cudaFree(dBuffer1)); checkCudaErrors(cudaFree(dBuffer2));
	checkCuSparseErrors(cusparseSpGEMM_destroyDescr(spgemmDesc));
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
void Sparse_mat<T>::convertCSRformatToDense(cusparseSpMatDescr_t matA, void** values) {

	// get buffer size and allocate buffer
	cusparseHandle_t handle;
	cusparseDnMatDescr_t matB;
	size_t bufferSize; void* buffer;
	// create handle
	checkCuSparseErrors(cusparseCreate(&handle));

	checkCuSparseErrors(cusparseSparseToDense_bufferSize(handle,
														 matA, matB,
														 CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
														 &bufferSize));
	checkCudaErrors(cudaMalloc(&buffer, bufferSize));

	// convert
	checkCuSparseErrors(cusparseSparseToDense(handle, matA, matB, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, buffer));

	// get dense format data
	checkCuSparseErrors(cusparseDnMatGetValues(matB, values));

	// free memory
	checkCudaErrors(cudaFree(buffer));
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
	convertCSRformatToDense(C.SpMatDescr, &C.data);
	return C;
}

// multiply and store the result in A - sparse-dense multiplication
template<typename T>
Sparse_mat<T>& Sparse_mat<T>::operator*=(const mat<T>& B) {
	Sparse_mat<T> D(M, B.N, memLocation::DEVICE);
	// create a C matrix for output
	T* C;
	checkCudaErrors(cudaMalloc((void**)&C, M * B.N * sizeof(T)));

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
	convertCSRformatToDense(C.SpMatDescr, &C.data);

	// copy the matrix to memory
	(HOST_ALLOC(memState) && N == B.N)
		? copyDeviceToHost(C.data, data, M, N) : copyDeviceToDevice(C.data, data, M, N);

	return *this;
}

/*
---------------------------------------------------------------------------------------------------
------------------------------VECTOR CLASS (MATRIX-VECTOR OPERATIONS)------------------------------
---------------------------------------------------------------------------------------------------
*/

template<typename T>
class nDvector {

};