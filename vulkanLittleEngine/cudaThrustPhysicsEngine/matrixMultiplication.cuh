#include "physicsDemoEngine.cuh"

// three memory maps
enum class memLocation { HOST, HOST_PINNED, DEVICE};

namespace matrix_h {

	// matrix class in column major order
	template<typename T>
	class mat {
	public:
		// thee matrix data
		T* m;
		// dimensions
		int M; int N; memLocation memState;

		mat(int n_rows, int n_cols, memLocation memLoc){
			// inputs
			this->M = n_rows;
			this->N = n_cols;
			this->memState = memLoc;

			// allocate memory
			if (memState == memLocation::HOST || memState == memLocation::HOST_PINNED) {
				allocateOnHost(m, M, N, memState);
			}
			else {
				allocateOnDevice(m, M, N);
			}
		}

		~mat() {
			(memState == memLocation::HOST || memState == memLocation::HOST_PINNED)
				? destroyHostMatrix(m, memState) : destroyDeviceMatrix(m);
		}

		// operators
		void operator+=(const mat<T>& B) {
			// allocate result matrix on device for cublas computation
			mat<T> C(M, N, memLocation::DEVICE);

			matAdd(CUBLAS_OP_N, CUBLAS_OP_N, M, N, this->m, M, this->memState, B.m, M, B.memState, C.m, M, C.memState);

			// copy back to this matrix
			(memState == memLocation::HOST || memState == memLocation::HOST_PINNED)
				? copyDeviceToHost(C.m, this->m, M, N) : copyDeviceToDevice(C.m, this->m, M, N);
		}

		// the + operator returns a device allocated matrix
		mat<T> operator+(const mat<T>& B) {
			// allocate result matrix on device for cublas computation
			mat<T> C(M, N, memLocation::DEVICE);

			matAdd(CUBLAS_OP_N, CUBLAS_OP_N, M, N, this->m, M, this->memState, B.m, M, B.memState, C.m, M, C.memState);

			// return the new matrix
			return C;
		}

		void operator*=(const mat<T>& B) {
			// allocate result matrix on device for cublas computation
			mat<T> C(M, B.N, memLocation::DEVICE);

			// C(M,N) = A(M,N)B(N,B.N)
			matMul(CUBLAS_OP_N, CUBLAS_OP_N, M, B.N, N, this->m, M, this->memState, B.m, N, B.memState, C.m, M , C.memState);

			// copy back to this matrix
			(memState == memLocation::HOST || memState == memLocation::HOST_PINNED)
				? copyDeviceToHost(C.m, this->m, M, N) : copyDeviceToDevice(C.m, this->m, M, N);
		}

		// the * operator returns a device allocated matrix
		mat<T> operator*(const mat<T>& B) {
			// allocate result matrix on device for cublas computation
			mat<T> C(M, N, memLocation::DEVICE);

			// C(M,N) = A(M,N)B(N,B.N)
			matMul(CUBLAS_OP_N, CUBLAS_OP_N, M, B.N, N, this->m, M, this->memState, B.m, N, B.memState, C.m, M, C.memState);

			// return the new matrix
			return C;
		}

		// copy host to device - from B to matrix m
		void operator<<=(const mat<T>& B) {
			copyHostToDevice(B.m, m, M, N);
		}

		// copy device to host - from B to matrix m
		void operator>>=(const mat<T>& B) {
			copyDeviceToHost(B.m, m, M, N);
		}

		// copy host to host - from B to matrix m
		void operator<=(const mat<T>& B) {
			copyHostToHost(B.m, m, M, N);
		}

		// copy device to device - from B to matrix m
		void operator>=(const mat<T>& B) {
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

		// memory and device management
		void allocateOnHost(T* m, int n_rows, int n_cols, memLocation memLoc);
		void allocateOnDevice(T* m, int n_rows, int n_cols);
		void copyHostToDevice(T* h_m, T* d_m, int n_rows, int n_cols);
		void copyHostToHost(T* h_m1, T* h_m2, int n_rows, int n_cols);
		void copyDeviceToHost(T* d_m, T* h_m, int n_rows, int n_cols);
		void copyDeviceToDevice(T* d_m1, T* d_m2, int n_rows, int n_cols);

		// check if transfer to device is needed
		bool checkToAllocateOnDevice(T* h_m, T* d_m, int n_rows, int n_cols, memLocation memLoc);

		// matrix addition
		void matAdd(cublasOperation_t transa,
					cublasOperation_t transb,
					int m, int n,
					const T* A, int lda,
					memLocation Aloc,
					const T* B, int ldb,
					memLocation Bloc,
					T* C, int ldc,
					memLocation Cloc);
		
		// matrix multiplication
		void matMul(cublasOperation_t transa,
					cublasOperation_t transb,
					int m, int n, int k,
					const T* A, int lda,
					memLocation Aloc,
					const T* B, int ldb,
					memLocation Bloc,
					T* C, int ldc,
					memLocation Cloc);

		// memory cleanup
		void destroyHostMatrix(T* m, memLocation memLoc);
		void destroyDeviceMatrix(T* m);
	};

	template<typename T> 
	void mat<T>::allocateOnHost(T* m, int n_rows, int n_cols, memLocation memLoc) {
		if (memLoc == memLocation::HOST_PINNED) {
			checkCudaErrors(cudaMallocHost((void**)&m, n_rows * n_cols * sizeof(T)));
		}
		else {
			m = (T*)malloc(n_rows * n_cols * sizeof(T));
		}
	}

	template<typename T>
	void mat<T>::allocateOnDevice(T* m, int n_rows, int n_cols) {
		checkCudaErrors(cudaMalloc((void**)&m, n_rows * n_cols * sizeof(T)));
	}

	template<typename T>
	void mat<T>::copyHostToDevice(T* h_m, T* d_m, int n_rows, int n_cols) {
		cudaStream_t stream;
		checkCudaErrors(cudaStreamCreate(&stream));
		checkCudaErrors(cudaMemcpyAsync(d_m, h_m, n_rows * n_cols * sizeof(T), cudaMemcpyHostToDevice, stream));
		checkCudaErrors(cudaStreamDestroy(stream));
	}

	template<typename T>
	void mat<T>::copyHostToHost(T* h_m1, T* h_m2, int n_rows, int n_cols) {
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
	void mat<T>::copyDeviceToDevice(T* d_m1, T* d_m2, int n_rows, int n_cols) {
		cudaStream_t stream;
		checkCudaErrors(cudaStreamCreate(&stream));
		checkCudaErrors(cudaMemcpyAsync(d_m2, d_m1, n_rows * n_cols * sizeof(T), cudaMemcpyDeviceToDevice, stream));
		checkCudaErrors(cudaStreamDestroy(stream));
	}

	template<typename T>
	bool mat<T>::checkToAllocateOnDevice(T* h_m, T* d_m, int n_rows, int n_cols, memLocation memLoc) {
		if (memLoc == memLocation::HOST || memLoc == memLocation::HOST_PINNED) {
			allocateOnDevice(d_m, n_rows, n_cols);
			copyHostToDevice(h_m, d_m, n_rows, n_cols);
			return true;
		}
		return false;
	}

	// wrapper to deal with all data types
	template<typename T>
	cublasStatus_t cublasGeam_wrapper(cublasHandle_t handle,
									   cublasOperation_t transa,
									   cublasOperation_t transb,
									   int m, int n,
									   const T* alpha,
									   const T* A, int lda,
									   const T* beta,
									   const T* B, int ldb,
									   T* C, int ldc) {
		if (typeid(T) == typeid(float)) {
			return cublasSgeam(handle,
							   transa,
							   transb,
							   m, n,
							   (const float*)alpha,
							   (const float*)A, lda,
							   (const float*)beta,
							   (const float*)B, ldb,
							   (float*)C, ldc);
		}
		else if (typeid(T) == typeid(double)) {
			return cublasDgeam(handle,
							   transa,
							   transb,
							   m, n,
							   (const double*)alpha,
							   (const double*)A, lda,
							   (const double*)beta,
							   (const double*)B, ldb,
							   (double*)C, ldc);
		}
		else if (typeid(T) == typeid(cuComplex)) {
			return cublasCgeam(handle,
							   transa,
							   transb,
							   m, n,
							   (const cuComplex*)alpha,
							   (const cuComplex*)A, lda,
							   (const cuComplex*)beta,
							   (const cuComplex*)B, ldb,
							   (cuComplex*)C, ldc);
		}
		else if (typeid(T) == typeid(cuDoubleComplex)) {
			return cublasZgeam(handle,
							   transa,
							   transb,
							   m, n,
							   (const cuDoubleComplex*)alpha,
							   (const cuDoubleComplex*)A, lda,
							   (const cuDoubleComplex*)beta,
							   (const cuDoubleComplex*)B, ldb,
							   (cuDoubleComplex*)C, ldc);
		}
		else {
			printf("type T is not supported for current matrix operation \n");
			return CUBLAS_STATUS_NOT_SUPPORTED;
		}
	}

	// C is defined to be allocated on device in matrix operators - no need for
	// memory Location checks
	template<typename T>
	void mat<T>::matAdd(cublasOperation_t transa,
						cublasOperation_t transb,
						int m, int n,
						const T* A, int lda,
						memLocation Aloc,
						const T* B, int ldb,
						memLocation Bloc,
						T* C, int ldc,
						memLocation Cloc) {

		const T alpha = static_cast<T>(1);
		const T beta = static_cast<T>(1);
		cublasHandle_t handle;


		// copy host data to device if needed
		T* d_A; T* d_B; bool A_status; bool B_status;
		A_status = checkToAllocateOnDevice(const_cast<T*>(A), d_A, lda, n, Aloc);
		B_status = checkToAllocateOnDevice(const_cast<T*>(B), d_B, ldb, n, Bloc);

		// create handle
		checkCuBLAS_status(cublasCreate_v2(&handle));

		// compute
		if (A_status && B_status) {
			checkCuBLAS_status(cublasGeam_wrapper<T>(handle, transa, transb, m, n, &alpha, d_A, lda, &beta, d_B, ldb, C, ldc));
			cudaFree(d_A); cudaFree(d_B);
		}
		else if (A_status) {
			checkCuBLAS_status(cublasGeam_wrapper<T>(handle, transa, transb, m, n, &alpha, d_A, lda, &beta, B, ldb, C, ldc));
			cudaFree(d_A);
		}
		else if (B_status) {
			checkCuBLAS_status(cublasGeam_wrapper<T>(handle, transa, transb, m, n, &alpha, A, lda, &beta, d_B, ldb, C, ldc));
			cudaFree(d_B);
		}
		else {
			checkCuBLAS_status(cublasGeam_wrapper<T>(handle, transa, transb, m, n, &alpha, A, lda, &beta, B, ldb, C, ldc));
		}

		checkCudaErrors(cudaDeviceSynchronize());

		// destroy handle
		checkCuBLAS_status(cublasDestroy_v2(handle));

	}

	// matrix multiplication wrapper to deal with all data types
	template<typename T>
	cublasStatus_t cublasGemm_wrapper(cublasHandle_t handle,
									  cublasOperation_t transa,
								      cublasOperation_t transb,
									  int m, int n, int k,
									  const T* alpha,
									  const T* A, int lda,
									  const T* beta,
									  const T* B, int ldb,
									  T* C, int ldc) {
		if (typeid(T) == typeid(float)) {
			return cublasSgemm_v2(handle,
								  transa,
								  transb,
								  m, n, k,
								  (const float*)alpha,
								  (const float*)A, lda,
								  (const float*)B, ldb,
								  (const float*)beta,
								  (float*)C, ldc);
		}
		else if (typeid(T) == typeid(double)) {
			return cublasDgemm_v2(handle,
								  transa,
								  transb,
								  m, n, k,
								  (const double*)alpha,
								  (const double*)A, lda,
								  (const double*)B, ldb,
								  (const double*)beta,
								  (double*)C, ldc);
		}
		else if (typeid(T) == typeid(cuComplex)) {
			return cublasCgemm_v2(handle,
								  transa,
								  transb,
								  m, n, k,
								  (const cuComplex*)alpha,
								  (const cuComplex*)A, lda,
								  (const cuComplex*)B, ldb,
								  (const cuComplex*)beta,
								  (cuComplex*)C, ldc);
		}
		else if (typeid(T) == typeid(cuDoubleComplex)) {
			return cublasZgemm_v2(handle,
								  transa,
								  transb,
								  m, n, k,
								  (const cuDoubleComplex*)alpha,
								  (const cuDoubleComplex*)A, lda,
								  (const cuDoubleComplex*)B, ldb,
								  (const cuDoubleComplex*)beta,
								  (cuDoubleComplex*)C, ldc);
		}
		else {
			printf("type T is not supported for current matrix operation \n");
			return CUBLAS_STATUS_NOT_SUPPORTED;
		}
	}

	// C is defined to be allocated on device in matrix operators - no need for
	// memory Location checks
	template<typename T>
	void mat<T>::matMul(cublasOperation_t transa,
						cublasOperation_t transb,
						int m, int n, int k,
						const T* A, int lda,
						memLocation Aloc,
						const T* B, int ldb,
						memLocation Bloc,
						T* C, int ldc,
						memLocation Cloc) {

		const T alpha = static_cast<T>(1);
		const T beta = static_cast<T>(0);
		cublasHandle_t handle;


		// copy host data to device if needed
		T* d_A; T* d_B; bool A_status; bool B_status;
		A_status = checkToAllocateOnDevice(const_cast<T*>(A), d_A, lda, n, Aloc);
		B_status = checkToAllocateOnDevice(const_cast<T*>(B), d_B, ldb, n, Bloc);

		// create handle
		checkCuBLAS_status(cublasCreate_v2(&handle));
		// compute
		if (A_status && B_status) {
			checkCuBLAS_status(cublasGemm_wrapper<T>(handle, transa, transb, m, n, k, &alpha, d_A, lda, &beta, d_B, ldb, C, ldc));
			cudaFree(d_A); cudaFree(d_B);
		}
		else if (A_status) {
			checkCuBLAS_status(cublasGemm_wrapper<T>(handle, transa, transb, m, n, k, &alpha, d_A, lda, &beta, B, ldb, C, ldc));
			cudaFree(d_A);
		}
		else if (B_status) {
			checkCuBLAS_status(cublasGemm_wrapper<T>(handle, transa, transb, m, n, k, &alpha, A, lda, &beta, d_B, ldb, C, ldc));
			cudaFree(d_B);
		}
		else {
			checkCuBLAS_status(cublasGemm_wrapper<T>(handle, transa, transb, m, n, k, &alpha, A, lda, &beta, B, ldb, C, ldc));
		}

		checkCudaErrors(cudaDeviceSynchronize());

		// destroy handle
		checkCuBLAS_status(cublasDestroy_v2(handle));

	}

	template<typename T>
	void mat<T>::destroyHostMatrix(T* m, memLocation memLoc) {
		if (memLoc == memLocation::HOST_PINNED) checkCudaErrors(cudaFreeHost(m));
		else free(m);
	}

	template<typename T>
	void mat<T>::destroyDeviceMatrix(T* m) {
		checkCudaErrors(cudaFree(m));
	}
}