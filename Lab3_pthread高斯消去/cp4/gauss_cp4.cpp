#include <iostream>
#include <cstdlib>
#include <tmmintrin.h>
#include <pthread.h>
#include <semaphore.h>
#include <string>
#include <fstream>
#include <sys/time.h>
#define millitime(x) (x.tv_sec * 1000 + x.tv_usec / 1000.0)
#define abs(x) ((x < 0) ? -(x) : (x))
#define MaxN 2048
using namespace std;

int n, thread_count;
string paths[4];
string path;
string file_num = "4";
float **A = NULL, **mother = NULL;
sem_t *sem_Division;
sem_t *sem_Elimination;
sem_t sem_leader;
pthread_barrier_t barrier;

void common_gauss() {
	for (int k = 0; k < n; k++) {
		float ele = A[k][k];
		for (int j = k + 1; j < n; j++)
			A[k][j] = A[k][j] / ele;
		A[k][k] = 1.0;
		for (int i = k + 1; i < n; i++) {
			for (int j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			A[i][k] = 0;
		}
	}
}

void *thread_func_sem(void *param) {
    long long t_id = (long long)param;
    for (int k = 0; k < n; k++) {
        if (t_id == 0) {
            float ele = A[k][k];
            for (int j = k + 1; j < n; j++)
                A[k][j] = A[k][j] / ele;
            A[k][k] = 1.0;
            for (int i = 0; i < thread_count - 1; i++)
                sem_post(sem_Division + i);
        }
        else
            sem_wait(sem_Division + t_id - 1);
        for (int i = k + t_id + 1; i < n; i += thread_count) {
			for (int j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			A[i][k] = 0;
        }
        if (t_id == 0) {
            for (int i = 0; i < thread_count - 1; i++)
                sem_wait(&sem_leader);
            for (int i = 0; i < thread_count - 1; i++)
                sem_post(sem_Elimination + i);
        }
        else {
            sem_post(&sem_leader);
            sem_wait(sem_Elimination + t_id - 1);
        }
    }
    pthread_exit(NULL);
    return NULL;
}

void* thread_func_sem_simd(void *param) {
    long long t_id = (long long)param;
    for (int k = 0; k < n; k++) {
        if (t_id == 0) {
            float ele = A[k][k];
            for (int j = k + 1; j < n; j++)
                A[k][j] = A[k][j] / ele;
            A[k][k] = 1.0;
            for (int i = 0; i < thread_count - 1; i++)
                sem_post(sem_Division + i);
        }
        else
            sem_wait(sem_Division + t_id - 1);
        int interval = ((n - k - 1) % thread_count == 0) ? (n - k - 1) / thread_count : (n - k - 1) / thread_count + 1;
        int upp_bound = min((long long)n, k + 1 + interval * (t_id + 1));
        for (int i = k + interval * t_id + 1; i < upp_bound; i++) {
            __m128 v0, v1, v2;
            v1 = _mm_set_ps(A[i][k], A[i][k], A[i][k], A[i][k]);
            int j = k + 1;
			for (j; j <= n - 4; j += 4) {
			    v2 = _mm_loadu_ps(A[k] + j);
                v0 = _mm_loadu_ps(A[i] + j);
                v2 = _mm_mul_ps(v1, v2);
                v0 = _mm_sub_ps(v0, v2);
				_mm_storeu_ps(A[i] + j, v0);
			}
			for (j; j < n; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			A[i][k] = 0;
        }
        if (t_id == 0) {
            for (int i = 0; i < thread_count - 1; i++)
                sem_wait(&sem_leader);
            for (int i = 0; i < thread_count - 1; i++)
                sem_post(sem_Elimination + i);
        }
        else {
            sem_post(&sem_leader);
            sem_wait(sem_Elimination + t_id - 1);
        }
    }
    pthread_exit(NULL);
    return NULL;
}

void static_gauss_sem() {
    sem_Division = new sem_t[thread_count - 1];
    sem_Elimination = new sem_t[thread_count - 1];
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < thread_count - 1; i++) {
        sem_init(sem_Division + i, 0, 0);
        sem_init(sem_Elimination + i, 0, 0);
    }
    pthread_t *handles = new pthread_t[thread_count];
    for (int i = 0; i < thread_count; i++)
        pthread_create(handles + i, NULL, thread_func_sem, (void *)(long long)i);
    for (int i = 0; i < thread_count; i++)
        pthread_join(handles[i], NULL);
    delete[] handles;
    for (int i = 0; i < thread_count - 1; i++) {
        sem_destroy(sem_Division + i);
        sem_destroy(sem_Elimination + i);
    }
    sem_destroy(&sem_leader);
    delete[] sem_Division;
    delete[] sem_Elimination;
}

void static_gauss_sem_simd() {
    sem_Division = new sem_t[thread_count - 1];
    sem_Elimination = new sem_t[thread_count - 1];
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < thread_count - 1; i++) {
        sem_init(sem_Division + i, 0, 0);
        sem_init(sem_Elimination + i, 0, 0);
    }
    pthread_t *handles = new pthread_t[thread_count];
    for (int i = 0; i < thread_count; i++)
        pthread_create(handles + i, NULL, thread_func_sem_simd, (void *)(long long)i);
    for (int i = 0; i < thread_count; i++)
        pthread_join(handles[i], NULL);
    delete[] handles;
    for (int i = 0; i < thread_count - 1; i++) {
        sem_destroy(sem_Division + i);
        sem_destroy(sem_Elimination + i);
    }
    sem_destroy(&sem_leader);
    delete[] sem_Division;
    delete[] sem_Elimination;
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
    paths[0] = "common_gauss";
    paths[1] = "static_gauss_sem";
    paths[2] = "static_gauss_sem_simd";
    void (*p[4])();
    p[0] = &common_gauss;
    p[1] = &static_gauss_sem;
    p[2] = &static_gauss_sem_simd;
 	n = 4;
	thread_count = atoi(argv[1]);
	while (n <= MaxN) {
    	init();
    	cout << "n: " << n << endl;
        for (int i = 0; i < 3; i++) {
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
