#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <windows.h>

__global__
void division_kernel(int row, float* A, int n) {
    int id = threadIdx.x + blockDim.x * blockIdx.x + row + 1;
    int stride = blockDim.x * gridDim.x;
    for (int i = id; i < n; i += stride) {
        A[row * n + i] /= A[row *n + row];
    }
    if (id == row + 1)
        A[row * n + row] = 1.0;
}

__global__
void eliminate_kernel(int row, float* A, int n) {
    int idX = threadIdx.x + blockIdx.x * blockDim.x + row + 1;
    int idY = threadIdx.y + blockIdx.y * blockDim.y + row + 1;
    int strideX = gridDim.x * blockDim.x;
    int strideY = gridDim.y * blockDim.y;
    for (int i = idX; i < n; i += strideX) {
        for (int j = idY; j < n; j += strideY) {
            A[i * n + j] -= A[i * n + row] * A[row * n + j];
        }
        if (idY == row + 1)
            A[i * n + row] = 0;
    }
}

void common_gauss(float* A, int n) {
	for (int k = 0; k < n; k++) {
		float ele = A[k * n + k];
		for (int j = k + 1; j < n; j++)
			A[k * n + j] = A[k * n + j] / ele;
		A[k * n + k] = 1.0;
		for (int i = k + 1; i < n; i++) {
			for (int j = k + 1; j < n; j++)
				A[i * n + j] = A[i * n + j] - A[i * n + k] * A[k * n + j];
			A[i * n + k] = 0;
		}
	}
}

float** mother;

void init(float *A, int n) {
    mother = new float*[n];
	for (int i = 0; i < n; i++) {
		mother[i] = new float[n];
		for (int j = 0; j < n; j++) {
            mother[i][j] = 0;
            A[i * n + j] = 0;
		}
	}
	for (int i = 0; i < n; i++)
		for (int j = i; j < n; j++)
			mother[i][j] = (j == i) ? 1 : i + j;
}

void arr_reset(float* A, int n) {
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			A[i * n + j] = mother[i][j];
	for (int i = 1; i < n; i++)
		for (int j = 0; j < i; j++)
			for (int k = 0; k < n; k++)
				A[i * n + k] += mother[j][k];
}

void printResult(float* A, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%f ", mother[i][j]);
		}
        printf("\n");
    }
    printf("\n");
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
            printf("%f ", A[i * n + j]);
        }
        printf("\n");
	}
}

int main(int argc, char* argv[]) {
    if (argc < 2)
        return 0;
    int deviceId;
    int numberOfSMs;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    int threadsPerBlock = 128;
    int numberOfBlocks = numberOfSMs * 16;
    dim3 threadsPerBlockDim3(16, 16, 1);
    dim3 numberOfBlocksDim3(numberOfSMs, numberOfSMs, 1);

    int n = atoi(argv[1]);
    float *A = NULL;
    size_t size = n * n * sizeof(float);
    cudaMallocManaged(&A, size);
    init(A, n);
    arr_reset(A, n);

    long long head, tail ,freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    common_gauss(A, n);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    printf("Common_alg time is: %f ms\n", (tail - head) * 1000.0 / freq);


    arr_reset(A, n);
    cudaEvent_t start, stop;
    float elapsedTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    cudaMemPrefetchAsync(A, size, deviceId);
	for (int k = 0; k < n; k++) {
        division_kernel<<<numberOfBlocks, threadsPerBlock>>>(k, A, n);
        cudaDeviceSynchronize();
		
        eliminate_kernel<<<numberOfBlocksDim3, threadsPerBlockDim3>>>(k, A, n);
	}
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU_alg time is: %f ms\n", elapsedTime);
    cudaFree(A);
    return 0;   
}