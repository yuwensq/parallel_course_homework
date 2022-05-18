#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include <string>
#include <fstream>
#include <omp.h>
#include <sys/time.h>
#define millitime(x) (x.tv_sec * 1000 + x.tv_usec / 1000.0)
#define abs(x) ((x < 0) ? -(x) : (x))
#define MaxN 2048
using namespace std;

int n, thread_count;
string paths[4];
string path;
string file_num = "1";
float **A = NULL, **mother = NULL;

void open_mp_default() {
    int i, j, k, ele;
    #pragma omp parallel num_threads(thread_count), private(i, j, k, ele)
	for (k = 0; k < n; k++) {
        #pragma omp single
        {
            ele = A[k][k];
            for (j = k + 1; j < n; j++)
                A[k][j] = A[k][j] / ele;
            A[k][k] = 1.0;
        }
        #pragma omp for
		for (i = k + 1; i < n; i++) {
			for (j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			A[i][k] = 0;
		}
	}
}

void open_mp_offload() {
    int i, j, k, ele;
	for (k = 0; k < n; k++) {
        {
            ele = A[k][k];
            for (j = k + 1; j < n; j++)
                A[k][j] = A[k][j] / ele;
            A[k][k] = 1.0;
        }
        #pragma omp target teams distribute parallel for \
                num_teams(8) map(tofrom:A) thread_limit(128)
		for (i = k + 1; i < n; i++) {
			for (j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			A[i][k] = 0;
		}
	}
}

void init() {
	A = new float*[n];
	mother = new float*[n];
	for (int i = 0; i < n; i++) {
		A[i] = new float[n];
		mother[i] = new float[n];
		for (int j = 0; j < n; j++) {
            A[i][j] = 0;
            mother[i][j] = 0;
		}
	}
	for (int i = 0; i < n; i++)
		for (int j = i; j < n; j++)
			mother[i][j] = (j == i) ? 1 : i + j;
}

void release_atrix() {
	for (int i = 0; i < n; i++) {
		delete[] A[i];
		delete[] mother[i];
	}
	delete[] A;
	delete[] mother;
}

void arr_reset() {
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			A[i][j] = mother[i][j];
	for (int i = 1; i < n; i++)
		for (int j = 0; j < i; j++)
			for (int k = 0; k < n; k++)
				A[i][k] += mother[j][k];
}

void printResult() {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << mother[i][j] << ' ';
		}
		cout << endl;
	}
	cout << endl;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << A[i][j] << ' ';
		}
		cout << endl;
	}
}

void testResult() {
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			if (abs(A[i][j] - mother[i][j]) >= 1e-6) {
				cout << "Something wrong!" << endl;
				cout << i << ' ' << j << ' ' << A[i][j] << ' ' << mother[i][j] << endl;
				exit(-1);
			}
}

void test(void (*f)()) {
	int counter = 0;
	timeval begin, start, finish, now;
	float milliseconds, single_time;
	milliseconds = 0;
	gettimeofday(&begin, NULL);
	gettimeofday(&now, NULL);
	while (millitime(now) - millitime(begin) < 20) {
		counter++;
		arr_reset();
		gettimeofday(&start, NULL);
		f();
		gettimeofday(&finish, NULL);
		milliseconds += (millitime(finish) - millitime(start));
		gettimeofday(&now, NULL);
	}
    testResult();
	single_time = milliseconds / counter;
	cout << counter << " " << single_time << endl;
	ofstream out_file(path, ios::app);
	out_file << single_time << ' ';
	out_file.close();
}

int main(int argc, char *argv[])
{
    paths[0] = "open_mp_default";
    paths[1] = "open_mp_offload";
    void (*p[4])();
    p[0] = &open_mp_default;
    p[1] = &open_mp_offload;
 	n = 4;
	thread_count = atoi(argv[1]);
	while (n <= MaxN) {
    	init();
    	cout << "n: " << n << endl;
        for (int i = 0; i < 2; i++) {
            path = paths[i] + file_num;
            cout << path << ": ";
            test(p[i]);
        }
    	release_atrix();
		if (n < 256)
		    	n *= 2;
		else
			n += 128;
	}
    return 0;
}


