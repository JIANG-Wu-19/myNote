#include <iostream>

using namespace std;

class Cache {
public:
    bool state = false;  // Cache块的状态，false表示空闲，true表示占用
    int value = -1;      // Cache块存储的数值
    int count = 0;       // Cache块的未使用时间计数
};

const int M = 4; // Cache块数 
Cache cache[M];
const int N = 12;// 测试页面数 
int walk_sort[] = {1,1,2,4,3,5,2,1,6,7,1,3};// 测试数据 
void up_cache();

int main() {
	up_cache();
}

void up_cache() {
    int i = 0;
    while (i < N) {
        int j = 0;
	   	// 是否已满 
	   	cout << endl;
	   	cout << "--------------p" << walk_sort[i] << "到达--------------" << endl;
	   	cout << endl;
        while (j < M) {
            if((cache[j].state == false) && (walk_sort[i] != cache[j].value)) {
                cout << "cache有空闲块,不考虑是否要置换..." << endl;
                cout << walk_sort[i] << "被装入cache...." << endl;
                //cache[j].value = walk_sort[i++]; 
                cache[j].value = walk_sort[i]; 
                cache[j].state = true;
                cache[j].count = 0;
                int kk = 0;
                
                for (int x = 0; x < M; x++) {
                    cout << "cache块" << x << ": " << cache[x].value << endl;
                }
                cout << endl;
                
                // 更新其它cache块没使用时间
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
                cout << walk_sort[i] << "命中!!!" << endl;
                
                for (int x = 0; x < M; x++) {
                    cout << "cache块" << x << ": " << cache[x].value << endl;
                }
                cout << endl;
                //i++; 
                int kk = 0;
                cache[j].count=0;
                //更新其它cache块没使用时间
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

		//cache已满 
        if (j == M) {
            cout << "cache已经满了,考虑是否置换..." << endl;
            cout << endl;
            int k = 0;

            while (k < M) {
                if (cache[k].value == walk_sort[i]) {
                    cout << endl;
			        cout << walk_sort[i] << "命中!!!" << endl;
  
                    for (int x = 0; x < M; x++) {
                        cout << "cache块" << x << ": " << cache[x].value << endl;
                    }
                    
                    //i++;
                    cache[k].count = 0;
                    int kk = 0;
                     
                    //更新其它cache块没使用时间
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
            
            //考虑置换哪一块 
            if (k == M) {
                int ii = 0;
                int t = 0;//要替换的cache块号.
                int max = cache[ii].count;
                ii++; 
                while (ii < M) {
                    if(cache[ii].count > max) {
                        max = cache[ii].count;
                        t = ii;
                    }
                    ii++;
                }
                //置换
                cout<<cache[t].value<<"被"<<walk_sort[i]<<"在cache的"<<t<<"号块置换..."<<endl;
                //cache[t].value=walk_sort[i++];
                cache[t].value=walk_sort[i];
                cache[t].count=0;
                
                for (int x = 0; x < M; x++) {
                    cout << "cache块" << x << ": " << cache[x].value << endl;
                }
                int kk = 0;                
                //更新其它cache块没使用时间
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

