#include <arm_neon.h>
#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#define N 1024
#define millitime(x) (x.tv_sec * 1000 + x.tv_usec / 1000.0)
using namespace std;

float** A = NULL;

void arr_reset(int n) {
	static int last_scale;
	if (A != NULL) {
		for (int i = 0; i < last_scale; i++)
			delete[] A[i];
		delete[] A;
	}
	last_scale = n;
	A = new float* [n];
	srand(time(0));
	for(int i = 0; i < n; i++) {
		A[i] = new float [n];
		for(int j = 0; j < i; j++)
			A[i][j] = 0;
		A[i][i] = 1.0;
		for(int j = i + 1; j < n; j++)
			A[i][j] = rand();
	}
	for(int k = 0; k < n; k++)
		for(int i = k + 1; i < n; i++)
			for(int j = 0; j < n; j++)
				A[i][j] += A[k][j];
}

void test(void (*f)(int), int scale){
    int counter = 0;
    timeval start, finish, now;
    float milliseconds, single_time;
	gettimeofday(&start, NULL);
	gettimeofday(&now, NULL);
    while (millitime(now) - millitime(start) < 10){
        counter++;
        f(scale);
		gettimeofday(&now, NULL);
    }
    gettimeofday(&finish, NULL);
	milliseconds = millitime(finish) - millitime(start);
    single_time = milliseconds / counter;
    cout << single_time << endl;
}

void common_gauss(int n) {
	for (int k = 0; k < n; k++) {
		float ele = A[k][k];
		for (int j = k + 1; j < n; j++)
			A[k][j] = A [k][j] / ele;
		A[k][k] = 1.0;
		for (int i = k + 1; i < n; i++) {
			for (int j = k + 1; j < n; j++) 
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			A[i][k] = 0;
		}
	}
}

void opt1_gauss(int n) {
	for (int k = 0; k < n; k++) {
		float ele = A[k][k];
		float32x4_t v1 = vmovq_n_f32(ele);
		float32x4_t v0;
		int l;
		for (l = k + 1; l <= n - 4; l += 4) {
			v0 = vld1q_f32(A[k] + l);
			v0 = vdivq_f32(v0, v1);
			vst1q_f32(A[k] + l, v0);
		}
		for (l; l < n; l++)
			A[k][l] = A[k][l] / ele;
		A[k][k] = 1.0;
		for (int i = k + 1; i < n; i++) {
			for (int j = k + 1; j < n; j++) 
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			A[i][k] = 0;
		}
	}	
}

void opt2_gauss(int n) {
	for (int k = 0; k < n; k++) {
		float ele = A[k][k];
		for (int j = k + 1; j < n; j++)
			A[k][j] = A [k][j] / ele;
		A[k][k] = 1.0;
		for (int i = k + 1; i < n; i++) {
			float32x4_t v1 = vmovq_n_f32(A[i][k]);
			float32x4_t v0, v2;
			int j;
			for (j = k + 1; j <= n - 4; j += 4) {
				v2 = vld1q_f32(A[k] + j);
				v0 = vld1q_f32(A[i] + j);
				v2 = vmulq_f32(v1, v2);
				v0 = vsubq_f32(v0, v2);
				vst1q_f32(A[i] + j, v0);
			}
			for (j; j < n; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			A[i][k] = 0;
		}
	}	
}

void opt1And2_gauss(int n) {
	for (int k = 0; k < n; k++) {
		float ele = A[k][k];
		float32x4_t v1 = vmovq_n_f32(ele);
		float32x4_t v0;
		int l;
		for (l = k + 1; l <= n - 4; l += 4) {
			v0 = vld1q_f32(A[k] + l);
			v0 = vdivq_f32(v0, v1);
			vst1q_f32(A[k] + l, v0);
		}
		for (l; l < n; l++)
			A[k][l] = A[k][l] / ele;
		A[k][k] = 1.0;
		for (int i = k + 1; i < n; i++) {
			v1 = vmovq_n_f32(A[i][k]);
			float32x4_t v2;
			int j;
			for (j = k + 1; j <= n - 4; j += 4) {
				v2 = vld1q_f32(A[k] + j);
				v0 = vld1q_f32(A[i] + j);
				v2 = vmulq_f32(v1, v2);
				v0 = vsubq_f32(v0, v2);
				vst1q_f32(A[i] + j, v0);
			}
			for (j; j < n; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			A[i][k] = 0;	
		}
	}		
}

int main()
{
    int scale = N;
    while (scale <= N) {
    	cout << "scale: " << scale << endl;
		arr_reset(scale);
		cout << "common_gauss: ";
    	test(common_gauss, scale);
    	cout << "opt1_gauss: ";
    	test(opt1_gauss, scale);
    	cout << "opt2_gauss: ";
    	test(opt2_gauss, scale);
    	cout << "opt1And2_gauss: ";
    	test(opt1And2_gauss, scale);
        scale *= 2;
        cout << endl;
    }
    return 0;
}
