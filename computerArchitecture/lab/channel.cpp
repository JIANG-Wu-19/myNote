#include<iostream>
#include "Class.h"

using namespace std;

const int DeviceNum = 4;
const int MemoryNum = 4;

int main() {
	//使用类声明;
	Device Dev[DeviceNum];
	Memory Mem[MemoryNum];
	Ch_mannager c;
	// ui user;
	int i;
	//init the memory;
	cout << "init the memory" << endl;
	Mem[0].SetMcontent("love");
	cout << "Mem0 :\t" << "\t" <<  Mem[0].GetMcontent() << endl;
	Mem[1].SetMcontent("channel");
	cout << "Mem1 :\t" << "\t" <<  Mem[1].GetMcontent() << endl;
	Mem[2].SetMcontent("middle");
	cout << "Mem2 :\t" << "\t" <<  Mem[2].GetMcontent() << endl;
	Mem[3].SetMcontent("house");
	cout << "Mem3 :\t" << "\t" <<  Mem[3].GetMcontent() << endl;
	
	cout << "init the Device" << endl;
	//init the Device;
	Dev[0].SetRequireTime(10);
	Dev[0].SetNum(0);
	Dev[0].Setpriority(4);
	Dev[1].SetRequireTime(20);
	Dev[1].SetNum(1);
	Dev[1].Setpriority(3);
	Dev[2].SetRequireTime(25);
	Dev[2].SetNum(2);
	Dev[2].Setpriority(2);
	Dev[3].SetRequireTime(40);
	Dev[3].SetNum(3);
	Dev[3].Setpriority(1);
	
	for(i = 0; i < DeviceNum; i++) {
		Dev[i].PrintDevice();
	}

	//begin io;
	c.run(0);
	cout << "any io device?\t" << "\tyes -> 1" << endl;
	c.run(1);
	c.memoryToDevice(Mem,Dev);
	c.run(2);
	
	system("pause");
	return 0;
}
