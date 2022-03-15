#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <fstream>
#define N (1 << 25)
#define millitime(x) (x.tv_sec * 1000 + x.tv_usec / 1000.0)
using namespace std;

double a[N];

void init(int n) {
	for (int i = 0; i < n; i++)
		a[i] = 1;
}

double optimization2_time(int n) {
	int lengthB = (n >> 1);
	double *b = new double[lengthB];
	for (int i = 0; i < lengthB; i++)
		b[i] = a[2 * i] + a[2 * i + 1];
	for (int stage = 2; stage <= lengthB; stage <<= 1) {
		int step = (stage >> 1);
		for (int i = 0; i < lengthB; i += stage) {
			b[i] += b[i + step];
		}
	}
	double sum = b[0];
	delete[] b; 
	return sum;
}

void test(double (*f)(int), int scale, ofstream *output_file){
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
	*output_file << "n = " << scale << endl;
    *output_file << counter << ' ' << milliseconds
    << endl << single_time << endl;
}

int main()
{
    ofstream output_file("tree.txt");
    init(N);
    int scale = 2;//测定固定规模时修改此处及循环条件即可
    while (scale <= N) {
    	test(optimization2_time, scale, &output_file);
		scale *= 2;
    }
    output_file.close();
    return 0;
}


