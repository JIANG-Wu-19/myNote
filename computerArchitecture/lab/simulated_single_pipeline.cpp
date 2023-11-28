#include <iostream>
#include <string>

using namespace std;

const int SPACE = 4;  // ���ܲ�����Ŀ
const int NUM = 5;    // ��Ҫ��ˮ����ĸ����ָ����Ŀ
const int TIME = NUM + SPACE - 1;  // �洢��ͬʱ��θ������ܲ�����ָ��ֵ
// ED����ײ� EA���Խ� MA��β���� NL�����
const string INSTRUCTIONS[] = {"NL", "MA", "EA", "ED"};
int ts[SPACE][TIME] = {0};  // ��ʼ��ʱ��ͼ

void print();  // ���ʱ��ͼ
void pipeline(int ts[SPACE][TIME]);  // ��ˮ����ָ��״̬ת���㷨

int main() {
    cout << "Pipeline begins" << endl
         << endl;
    pipeline(ts);
    print();

    cout << endl
         << "Pipeline ends" << endl
         << endl;
    cout << "The Through Put of the pipeline is " << (double)NUM / TIME << "��t" << endl;
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
    int tempSpace = 0;  // ��¼�����ָ���
    int tempTime = 0;   // ��¼ʱ����ı仯
    for (int s = SPACE - 1; s >= 0; s--) {
        tempSpace = 1;
        for (int t = tempTime; t < TIME; t++) {
            ts[s][t] = tempSpace++;
        }
        tempTime++;
    }
}

