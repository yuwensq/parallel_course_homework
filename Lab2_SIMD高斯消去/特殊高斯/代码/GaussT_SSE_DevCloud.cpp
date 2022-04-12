#include <tmmintrin.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <unordered_map>
#include <cstdlib>
#include <sys/time.h>
#define millitime(x) (x.tv_sec * 1000 + x.tv_usec / 1000.0)
#define COLS 8399
#define NUMR 6375
#define NUME 4535
using namespace std;

const int totalCols = COLS / 32 + (COLS % 32 != 0 ? 1 : 0);
const int totalRows = NUMR + NUME; //总行数
int** A = NULL; //二维位图存储消元子和消元行 
int* lp = NULL; //存储每一行的最大首项
unordered_map<int, int> lpE2R; //首项对行数映射表

void initMartix() {
	if (A != NULL) {
		for (int i = 0; i < totalRows; i++)
			delete[] A[i];
		delete[] A;
		delete[] lp;
	}
	A = new int* [totalRows];
	for (int i = 0; i < totalRows; i++) {
		A[i] = new int[totalCols];
		for (int j = 0; j < totalCols; j++)
			A[i][j] = 0;
	}
	lp = new int[totalRows];
	for (int i = 0; i < totalRows; i++)
		lp[i] = -1;     //-1表示这一行寄了
	lpE2R.clear();
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
			A[nowRow][pos / 32] |= (1 << (pos % 32));
			lp[nowRow] = max(pos, lp[nowRow]);
		}
		if (nowRow >= NUME) {
			if (lpE2R.find(lp[nowRow]) != lpE2R.end()) {
				cout << "nop" << lpE2R[lp[nowRow]] << ' ' << nowRow - NUME << endl;
				exit(-1);
			}
			lpE2R[lp[nowRow]] = nowRow;
		}
		nowRow++;
		ss.clear();
	}
	input.close();
}

void readData() {
	int nowRow = 0;
	string s1 = "/home/u149651/test/2.txt";
	string s2 = "/home/u149651/test/1.txt";
	readFile(s1, nowRow);
	readFile(s2, nowRow);
}

void reset() {
	initMartix();
	readData();
}

void Gauss() {
	for (int i = 0; i < NUME; i++) {
		while (lp[i] > -1) {
			if (!(lpE2R.find(lp[i]) == lpE2R.end())) {
				int rowR = lpE2R[lp[i]];
				bool lpHasChanged = false;
				int p = totalCols - 1;
				for (p; p >= 0; p--) {
					A[i][p] ^= A[rowR][p];
				}
				for (p = totalCols - 1; p >= 3; p -= 4) {
					int x = ((A[i][p] | A[i][p - 1]) | (A[i][p - 2] | A[i][p - 3]));
					if (x == 0)
						continue;
					break;
				}
				for (p; p >= 0; p--) {
					if (A[i][p] != 0) {
						lpHasChanged = true;
						for (int k = 31; k >= 0; k--) {
							if ((A[i][p] & (1 << k)) != 0) {
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

void Gauss2() {
	for (int i = 0; i < NUME; i++) {
		while (lp[i] > -1) {
			if (!(lpE2R.find(lp[i]) == lpE2R.end())) {
				int rowR = lpE2R[lp[i]];
				bool lpHasChanged = false;
				__m128 v0, v1;
				int p = 0;
				for (p; p <= totalCols - 4; p += 4) {
					v0 = _mm_load_ps((float*)(A[i] + p));
					v1 = _mm_load_ps((float*)(A[rowR] + p));
					v0 = _mm_xor_ps(v0, v1);
					_mm_store_ps((float*)(A[i] + p), v0);
				}
				for (p; p < totalCols; p++)
					A[i][p] ^= A[rowR][p];
				for (p = totalCols - 1; p >= 3; p -= 4) {
					int x = ((A[i][p] | A[i][p - 1]) | (A[i][p - 2] | A[i][p - 3]));
					if (x == 0)
						continue;
					break;
				}
				for (p; p >= 0; p--) {
					if (A[i][p] != 0) {
						lpHasChanged = true;
						for (int k = 31; k >= 0; k--) {
							if ((A[i][p] & (1 << k)) != 0) {
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

void printResult() {//输出结果
        for (int i = 0; i < NUME; i++) {
	if (lp[i] == -1) {
		cout << endl;
		continue;
	}
	for (int j = totalCols - 1; j >= 0; j--) {
		for (int k = 31; k >= 0; k--) {
			if ((A[i][j] & (1 << k)) != 0) {
				cout << j * 32 + k << " ";
			}
		}
	}
	cout << endl;
}
}

void resultExam() {//验证代码是否正确
	ifstream inputResult("/home/u149651/test/3.txt");
	string line;
	stringstream ss;
	for (int i = 0; i < NUME; i++) {
		getline(inputResult, line);
		if (lp[i] == -1) {
			if (line.length() > 1) {
				cout << "wrong";
				exit(-1);
			}
			continue;
		}
		ss.clear();
		ss << line;
		int pos;
		for (int j = totalCols - 1; j >= 0; j--) {
			for (int k = 31; k >= 0; k--) {
				if ((A[i][j] & (1 << k)) != 0) {
					ss >> pos;
					if (pos != j * 32 + k) {
						cout << "wrong" << endl;
						exit(-1);
					}
				}
			}
		}
	}
	cout << "success" << endl;
	inputResult.close();
}

void test(void (*f)()) {
	int counter = 0;
	timeval begin, start, finish, now;
	float milliseconds, wastms, single_time;
	milliseconds = 0;
	wastms = 0;
	gettimeofday(&begin, NULL);
	gettimeofday(&now, NULL);
	while (millitime(now) - millitime(begin) < 20) {
		counter++;
		reset();
		gettimeofday(&start, NULL);
		f();
		gettimeofday(&finish, NULL);
		milliseconds += (millitime(finish) - millitime(start));
		gettimeofday(&now, NULL);
	}
	single_time = milliseconds / counter;
	cout << counter << " " << single_time << endl;
}

int main() {
	test(Gauss);
	//printResult();
	resultExam();
	test(Gauss2);
	resultExam();
	return 0;
}
