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

double ordinary_time(int n) {
	double sum = 0.0;
	for (int i = 0; i < n; i++)
		sum += a[i];
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
    ofstream output_file("linear.txt");
    init(N);
    int scale = 2;//�ⶨ�̶���ģʱ�޸Ĵ˴���ѭ����������
    while (scale <= N) {
    	test(ordinary_time, scale, &output_file);
		scale *= 2;
    }
    output_file.close();
    return 0;
}
