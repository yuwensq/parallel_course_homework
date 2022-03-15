#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <fstream>
#define N 12000
#define millitime(x) (x.tv_sec * 1000 + x.tv_usec / 1000.0)
using namespace std;

double **arr, vect[N], sum[N];

void init(int n) {
	arr = new double*[n];
    for (int i = 0; i < n; i++) {
		arr[i] = new double[n];
        for (int j = 0; j < n; j++) {
            arr[i][j] = i + j;
        }
        vect[i] = i;
    }
}

void release_mem(int n){
	for (int i = 0; i < n; i++)
		delete[] arr[i];
	delete[] arr;
}

void ordinary_time(int n) {
	for (int i = 0; i < n; i++)
		sum[i] = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            sum[i] += arr[j][i] * vect[j];
    }
}

void test(void (*f)(int), int scale, ofstream *output_file, ofstream *draw_picture){
    int counter = 0;
    timeval start, finish, now;
    float milliseconds, single_time;
	gettimeofday(&start, NULL);
	gettimeofday(&now, NULL);
	*output_file << "n = " << scale << endl;
    while (millitime(now) - millitime(start) < 10){
        counter++;
        f(scale);
		gettimeofday(&now, NULL);
    }
    gettimeofday(&finish, NULL);
	milliseconds = millitime(finish) - millitime(start);
    single_time = milliseconds / counter;
    *output_file << counter << ' ' << milliseconds
    << ' ' << single_time << endl ;
    *draw_picture << single_time << endl;
}

int main()
{
    ofstream output_result("ordinary_data.txt");
    ofstream output_drp("draw_pictureo.txt");
    int scale = 50; //测定固定规模时修改此处及循环条件即可
    while (scale <= N) {
		init(scale);
    	test(ordinary_time, scale, &output_result, &output_drp);
		release_mem(scale);
        scale += 50;
    }
    output_result.close();
    output_drp.close();
    return 0;
}
