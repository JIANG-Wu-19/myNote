#include <iostream>

using namespace std;

class Cache {
public:
    bool state = false;  // Cache���״̬��false��ʾ���У�true��ʾռ��
    int value = -1;      // Cache��洢����ֵ
    int count = 0;       // Cache���δʹ��ʱ�����
};

const int M = 4; // Cache���� 
Cache cache[M];
const int N = 12;// ����ҳ���� 
int walk_sort[] = {1,1,2,4,3,5,2,1,6,7,1,3};// �������� 
void up_cache();

int main() {
	up_cache();
}

void up_cache() {
    int i = 0;
    while (i < N) {
        int j = 0;
	   	// �Ƿ����� 
	   	cout << endl;
	   	cout << "--------------p" << walk_sort[i] << "����--------------" << endl;
	   	cout << endl;
        while (j < M) {
            if((cache[j].state == false) && (walk_sort[i] != cache[j].value)) {
                cout << "cache�п��п�,�������Ƿ�Ҫ�û�..." << endl;
                cout << walk_sort[i] << "��װ��cache...." << endl;
                //cache[j].value = walk_sort[i++]; 
                cache[j].value = walk_sort[i]; 
                cache[j].state = true;
                cache[j].count = 0;
                int kk = 0;
                
                for (int x = 0; x < M; x++) {
                    cout << "cache��" << x << ": " << cache[x].value << endl;
                }
                cout << endl;
                
                // ��������cache��ûʹ��ʱ��
                while (kk < M) {
                    if (kk != j && cache[kk].value != -1) {
                        cache[kk].count++;
                    }
                    kk++;
                }
                break; 
            }
            
            if (cache[j].value == walk_sort[i]) {
                cout << endl;
                cout << walk_sort[i] << "����!!!" << endl;
                
                for (int x = 0; x < M; x++) {
                    cout << "cache��" << x << ": " << cache[x].value << endl;
                }
                cout << endl;
                //i++; 
                int kk = 0;
                cache[j].count=0;
                //��������cache��ûʹ��ʱ��
                while (kk < M) {
                    if (kk != j && cache[kk].value != -1) {
                        cache[kk].count++;
                    }
                    kk++;
                }
                break;
            }
            j++;
        }

		//cache���� 
        if (j == M) {
            cout << "cache�Ѿ�����,�����Ƿ��û�..." << endl;
            cout << endl;
            int k = 0;

            while (k < M) {
                if (cache[k].value == walk_sort[i]) {
                    cout << endl;
			        cout << walk_sort[i] << "����!!!" << endl;
  
                    for (int x = 0; x < M; x++) {
                        cout << "cache��" << x << ": " << cache[x].value << endl;
                    }
                    
                    //i++;
                    cache[k].count = 0;
                    int kk = 0;
                     
                    //��������cache��ûʹ��ʱ��
                    while (kk < M) {
                        if (kk != k){
                            cache[kk].count++;
                        }
                        kk++;
                    }
                    break;
                }
                k++;
            } 
            
            //�����û���һ�� 
            if (k == M) {
                int ii = 0;
                int t = 0;//Ҫ�滻��cache���.
                int max = cache[ii].count;
                ii++; 
                while (ii < M) {
                    if(cache[ii].count > max) {
                        max = cache[ii].count;
                        t = ii;
                    }
                    ii++;
                }
                //�û�
                cout<<cache[t].value<<"��"<<walk_sort[i]<<"��cache��"<<t<<"�ſ��û�..."<<endl;
                //cache[t].value=walk_sort[i++];
                cache[t].value=walk_sort[i];
                cache[t].count=0;
                
                for (int x = 0; x < M; x++) {
                    cout << "cache��" << x << ": " << cache[x].value << endl;
                }
                int kk = 0;                
                //��������cache��ûʹ��ʱ��
                while (kk < M) {
                    if (kk != t) {
                        cache[kk].count++;
                    }
                    kk++;
                }
            }
        }
    	i++;
	}
}

