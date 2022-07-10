#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <unordered_map>
#include <cstdlib>
#include <Windows.h>
#define millitime(x) (x.tv_sec * 1000 + x.tv_usec / 1000.0)
#define COLS 8399 
#define NUMR 6375 
#define NUME 4535 
using namespace std;

const int totalCols = COLS / 32 + (COLS % 32 != 0 ? 1 : 0);
const int totalRows = NUMR + NUME; 
int* A = NULL; 
int* lp = NULL; 
unordered_map<int, int> lpE2R; 
size_t sizeT;

void init() {
    sizeT = totalCols * totalRows * sizeof(int);
    cudaMallocManaged(&A, sizeT);
	for (int i = 0; i < totalRows; i++) {
		for (int j = 0; j < totalCols; j++)
			A[i * totalCols + j] = 0;
	}
	lp = new int[totalRows];
	for (int i = 0; i < totalRows; i++)
		lp[i] = -1;     
	lpE2R.clear();
}

void release() {
    cudaFree(A);
    delete[] lp;
}

void readFile(string path, int& nowRow) {
	string line;
	stringstream ss;
	ss.clear();
	ifstream input(path);
	while (getline(input, line)) {
		int pos;
		ss << line;
		while (ss >> pos) {
			A[nowRow * totalCols + pos / 32] |= (1 << (pos % 32));
			lp[nowRow] = max(pos, lp[nowRow]);
		}
		if (nowRow >= NUME) {
			if (lpE2R.find(lp[nowRow]) != lpE2R.end()) {
                printf("Error in read");
				exit(-1);
			}
			lpE2R[lp[nowRow]] = nowRow;
		}
		nowRow++;
		ss.clear();
	}
	input.close();
}

void reset() {
    for (int i = 0; i < totalRows; i++) {
        for (int j = 0; j < totalCols; j++) {
            A[i * totalCols + j] = 0;
        }
    }
    for (int i = 0; i < totalRows; i++) {
        lp[i] = -1;
    }
    lpE2R.clear();
	int nowRow = 0;
	string s1 = "C:\\Users\\Lenovo\\Desktop\\test\\2.txt";
	string s2 = "C:\\Users\\Lenovo\\Desktop\\test\\1.txt";
	readFile(s1, nowRow);
	readFile(s2, nowRow);
}

void Gauss() {
	for (int i = 0; i < NUME; i++) {
		while (lp[i] > -1) {
			if (!(lpE2R.find(lp[i]) == lpE2R.end())) {
				int rowR = lpE2R[lp[i]];
				bool lpHasChanged = false;
				int p = totalCols - 1;
				for (p; p >= 0; p--) {
					A[i * totalCols + p] ^= A[rowR * totalCols + p];
				}
				for (p = totalCols - 1; p >= 3; p -= 4) {
					int x = ((A[i * totalCols + p] | A[i * totalCols + p - 1]) | (A[i * totalCols + p - 2] | A[i * totalCols + p - 3]));
					if (x == 0)
						continue;
					break;
				}
				for (p; p >= 0; p--) {
					if (A[i * totalCols + p] != 0) {
						lpHasChanged = true;
						for (int k = 31; k >= 0; k--) {
							if ((A[i * totalCols + p] & (1 << k)) != 0) {
								lp[i] = p * 32 + k;
								break;
							}
						}
						break;
					}

				}
				if (lpHasChanged == false)
					lp[i] = -1;
			}
			else {
				lpE2R[lp[i]] = i;
				break;
			}
		}
	}
}

__global__
void eliminate(int row, int row_r, int* a, int cols) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = id; i < cols; i += stride) {
        a[row * cols + i] ^= a[row_r * cols + i];
    }
}

void Gauss2() {
    int deviceId;
    int numberOfSMs;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    int threadsPerBlock = 64;
    int numberOfBlocks = numberOfSMs;
	for (int i = 0; i < NUME; i++) {
		while (lp[i] > -1) {
			if (!(lpE2R.find(lp[i]) == lpE2R.end())) {
				int rowR = lpE2R[lp[i]];
				bool lpHasChanged = false;
    			cudaMemPrefetchAsync(A, sizeT, deviceId);
                eliminate<<<numberOfBlocks, threadsPerBlock>>>(i, rowR, A, totalCols);
                cudaDeviceSynchronize();
				int p;
				for (p = totalCols - 1; p >= 3; p -= 4) {
					int x = ((A[i * totalCols + p] | A[i * totalCols + p - 1]) | (A[i * totalCols + p - 2] | A[i * totalCols + p - 3]));
					if (x == 0)
						continue;
					break;
				}
				for (p; p >= 0; p--) {
					if (A[i * totalCols + p] != 0) {
						lpHasChanged = true;
						for (int k = 31; k >= 0; k--) {
							if ((A[i * totalCols + p] & (1 << k)) != 0) {
								lp[i] = p * 32 + k;
								break;
							}
						}
						break;
					}

				}
				if (lpHasChanged == false)
					lp[i] = -1;
			}
			else {
				lpE2R[lp[i]] = i;
				break;
			}
		}
	}
}

void printResult() {
    for (int i = 0; i < NUME; i++) {
	if (lp[i] == -1) {
        printf("\n");
		continue;
	}
	for (int j = totalCols - 1; j >= 0; j--) {
		for (int k = 31; k >= 0; k--) {
			if ((A[i * totalCols + j] & (1 << k)) != 0) {
                printf("%d ", j * 32 + k);
			}
		}
	}
    printf("\n");
    }
}

void resultExam() {
	ifstream inputResult("C:\\Users\\Lenovo\\Desktop\\test\\3.txt");
	string line;
	stringstream ss;
	for (int i = 0; i < NUME; i++) {
		getline(inputResult, line);
		if (lp[i] == -1) {
			if (line.length() > 1) {
                printf("Wrong in test result");
				exit(-1);
			}
			continue;
		}
		ss.clear();
		ss << line;
		int pos;
		for (int j = totalCols - 1; j >= 0; j--) {
			for (int k = 31; k >= 0; k--) {
				if ((A[i * totalCols + j] & (1 << k)) != 0) {
					ss >> pos;
					if (pos != j * 32 + k) {
                        printf("Wrong in test result");
						exit(-1);
					}
				}
			}
		}
	}
    printf("Success\n");
	inputResult.close();
}

void test(void (*f)()) {
	reset();
    long long head, tail ,freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
	f();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    printf("time is: %lf ms\n", (tail - head) * 1000.0 / freq);
}

int main() {
    init();
	test(Gauss);
	//printResult();
	resultExam();
	test(Gauss2);
	resultExam();
    release();
	return 0;
}