#include <vector>
#include <pthread.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <semaphore.h>
#include <pthread.h>
#include <unordered_map>
#include <cstdlib>
#include <sys/time.h>
#define millitime(x) (x.tv_sec * 1000 + x.tv_usec / 1000.0)
using namespace std;

int thread_count;
string paths[3];
string path;
string file_num = "4";
vector<string> file_names;
string file_name;
int COLS, NUMR, NUME;
int totalCols;
int totalRows;
int** A = NULL; //二维位图存储消元子和消元行
int* lp = NULL; //存储每一行的最大首项
unordered_map<int, int> lpE2R; //首项对行数映射表

bool work_over = false;
pthread_barrier_t barrier1, barrier2, barrier3, barrier4;


void release_mem() {
	if (A != NULL) {
		for (int i = 0; i < totalRows; i++)
			delete[] A[i];
		delete[] A;
		delete[] lp;
		A = NULL;
		lp = NULL;
	}
}

void initMartix() {
	A = new int* [totalRows];
	for (int i = 0; i < totalRows; i++) {
		A[i] = new int[totalCols];
		for (int j = 0; j < totalCols; j++)
			A[i][j] = 0;
	}
	lp = new int[totalRows];
	for (int i = 0; i < totalRows; i++)
		lp[i] = -1;	//-1表示这一行寄了
	lpE2R.clear();
}

void fresh_arg() {
    string now_file_name = file_name;
	for (int i = 0; i < now_file_name.size(); i++)
		if (now_file_name[i] == '_')
			now_file_name[i] = ' ';
	stringstream ss;
	ss << now_file_name;
	ss >> COLS;
	ss >> COLS;
	ss >> NUMR;
	ss >> NUME;
	totalCols = COLS / 32 + (COLS % 32 != 0 ? 1 : 0);
	totalRows = NUMR + NUME;
}

void readFile(string path, int &nowRow) {
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
				cout << "nop" << lpE2R[lp[nowRow]] << ' ' << nowRow - NUME <<endl;
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
	readFile("/home/s2010234/test/" + file_name + "/2.txt", nowRow);
	readFile("/home/s2010234/test/" + file_name + "/1.txt", nowRow);
}

void reset() {
	for (int i = 0; i < totalRows; i++) {
		for (int j = 0; j < totalCols; j++)
			A[i][j] = 0;
	}
	for (int i = 0; i < totalRows; i++)
		lp[i] = -1;	//-1表示这一行寄了
	lpE2R.clear();
	readData();
}

void common_gausst() {
	for (int i = 0; i < NUME; i++) {
		while (lp[i] > -1) {
			if (!(lpE2R.find(lp[i]) == lpE2R.end())) {
				int rowR = lpE2R[lp[i]];
				bool lpHasChanged = false;
				int p;
				for (p = 0; p < totalCols; p++) {
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

void *thread_func(void *param) {
	long long t_id = (long long)param;
    for (int i = 0; i < NUME; i++) {
		while (lp[i] > -1) {
			if (!(lpE2R.find(lp[i]) == lpE2R.end())) {
				int rowR = lpE2R[lp[i]];
				int p;
				for (p = t_id; p < totalCols; p += thread_count) {
					A[i][p] ^= A[rowR][p];
				}
                pthread_barrier_wait(&barrier1);
				if (t_id == 0) {
                    bool lpHasChanged = false;
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
                pthread_barrier_wait(&barrier2);
            }
			else {
                pthread_barrier_wait(&barrier4);
                if (t_id == 0)
                    lpE2R[lp[i]] = i;
                pthread_barrier_wait(&barrier3);
				break;
			}
		}
	}
	return NULL;
}

void static_gausst_p() {
    pthread_barrier_init(&barrier1, NULL, thread_count);
    pthread_barrier_init(&barrier2, NULL, thread_count);
    pthread_barrier_init(&barrier3, NULL, thread_count);
    pthread_barrier_init(&barrier4, NULL, thread_count);
    pthread_t *handles = new pthread_t[thread_count];
    for (int i = 0; i < thread_count; i++)
        pthread_create(handles + i, NULL, thread_func, (void *)(long long)i);
	for (int i = 0; i < thread_count; i++)
        pthread_join(handles[i], NULL);
    delete[] handles;
    pthread_barrier_destroy(&barrier1);
    pthread_barrier_destroy(&barrier2);
    pthread_barrier_destroy(&barrier3);
    pthread_barrier_destroy(&barrier4);
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
	ifstream inputResult("/home/s2010234/test/"+ file_name +"/3.txt");
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
	float milliseconds, wastms,single_time;
	milliseconds = 0;
	wastms = 0;
	gettimeofday(&begin, NULL);
	gettimeofday(&now, NULL);
	while (millitime(now) - millitime(begin) < 100) {
		counter++;
		reset();
		gettimeofday(&start, NULL);
		f();
		gettimeofday(&finish, NULL);
		milliseconds += (millitime(finish) - millitime(start));
		gettimeofday(&now, NULL);
	}
	resultExam();
	single_time = milliseconds / counter;
	cout << file_name << " " << path << " " <<counter << " " << single_time << endl;
	ofstream out_file(path, ios::app);
	out_file << single_time << ' ';
	out_file.close();
}

int main(int argc, char *argv[]) {
    thread_count = atoi(argv[1]);
    file_names.push_back("1_130_22_8");
    file_names.push_back("2_254_106_53");
    file_names.push_back("3_562_170_53");
    file_names.push_back("4_1011_539_263");
    file_names.push_back("5_2362_1226_453");
    file_names.push_back("6_3799_2759_1953");
	file_names.push_back("7_8399_6375_4535");
    paths[0] = "common_gausst";
    paths[1] = "static_gausst_p";
    void (*p[3])();
    p[0] = &common_gausst;
    p[1] = &static_gausst_p;
    for (int i = 0; i < file_names.size(); i++) {
        file_name = file_names[i];
        fresh_arg();
        initMartix();
        for (int j = 0; j < 2; j++) {
            path = paths[j] + file_num;
            test(p[j]);
        }
        release_mem();
    }
	return 0;
}



