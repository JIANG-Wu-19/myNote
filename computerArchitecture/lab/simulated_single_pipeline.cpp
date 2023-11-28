#include <iostream>
#include <string>

using namespace std;

const int SPACE = 4;  // 功能部件数目
const int NUM = 5;    // 需要流水处理的浮点加指令数目
const int TIME = NUM + SPACE - 1;  // 存储不同时间段各个功能部件内指令值
// ED：求阶差 EA：对阶 MA：尾数加 NL：规格化
const string INSTRUCTIONS[] = {"NL", "MA", "EA", "ED"};
int ts[SPACE][TIME] = {0};  // 初始化时空图

void print();  // 输出时空图
void pipeline(int ts[SPACE][TIME]);  // 流水线中指令状态转换算法

int main() {
    cout << "Pipeline begins" << endl
         << endl;
    pipeline(ts);
    print();

    cout << endl
         << "Pipeline ends" << endl
         << endl;
    cout << "The Through Put of the pipeline is " << (double)NUM / TIME << "Δt" << endl;
    cout << "The Speedup of the pipeline is " << ((double)NUM * SPACE) / TIME << endl;
    cout << "The Efficiency of the pipeline is " << ((double)NUM * SPACE) / (TIME * SPACE) << endl;

    return 0;
}

void print() {
    for (int i = 0; i < TIME; ++i) {
        cout << "After time slice " << i + 1 << endl;
        for (int j = 0; j < SPACE; ++j) {
            if (i < NUM && ts[j][i] == 0) {
                cout << endl;
            } else {
                for (int k = 0; k < i + 1; ++k) {
                    if (ts[j][k] != 0) {
                        cout << INSTRUCTIONS[j] << ts[j][k];
                    }
                    cout << "\t";
                }
                cout << endl;
            }
        }
    }
}

void pipeline(int ts[SPACE][TIME]) {
    int tempSpace = 0;  // 记录处理的指令号
    int tempTime = 0;   // 记录时间轴的变化
    for (int s = SPACE - 1; s >= 0; s--) {
        tempSpace = 1;
        for (int t = tempTime; t < TIME; t++) {
            ts[s][t] = tempSpace++;
        }
        tempTime++;
    }
}

