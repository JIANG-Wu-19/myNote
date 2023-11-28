#include<iostream>
#include<string.h>
using namespace std;
const int NONE = 0;
const int INIT = 1;
const int FINISH = 2;
//IO设备类;
class Device {
	private:
		int Num;
		int RequireTime;
		int priority;
		char* Content;
		int itslen;
		int RequireState;// 0->r 1-> nr
		int itspos;
	public:
		Device();
		Device(int itsNum,int itsRequireTime);
		~Device() {
			delete[] Content;
		}
		void SetNum(int itsNum);
		int GetNum() {
			return Num;
		}
		void SetRequireTime(int itsRequireTime);
		int GetRequireTime() {
			return RequireTime;
		}
		void Setpriority(int itspriority);
		int Getpriority() {
			return priority;
		}
		void SetContent(const char* itscontent);
		char* GetContent() const {
			return this->Content;
		}
		void SetRequireState(int state);
		int GetRequireState() {
			return RequireState;
		}
		void PrintDevice();
// int operator++(){ return itspos++; };
		int Getpos() {
			return itspos;
		}
};

Device::Device():
	Num(0),
	RequireTime(0),
	Content(NULL),
	RequireState(0),
	priority(0),
	itspos(0) {
}

Device::Device(int itsNum,int itsRequireTime):
	Num(itsNum),
	RequireTime(itsRequireTime) {
}

void Device::SetNum(int itsNum) {
	Num = itsNum;
}

void Device::SetRequireTime(int itsRequireTime) {
	RequireTime = itsRequireTime;
}

void Device::Setpriority(int itspriority) {
	priority = itspriority;
}

void Device::SetContent(const char* itsContent) {
	if (!this->Content) {
		itslen = strlen(itsContent);
		Content = new char[itslen + 1];
	}

	Content[itspos] = itsContent[itspos];
	itspos++;
	Content[itspos] = '\0';
}

void Device::SetRequireState(int state) {
	RequireState = state ;//0 = f, 1 = t;
}

void Device::PrintDevice() {
	cout << "设备" << Num << "\t请求时间" << RequireTime << "\t内容:" << (Content ? Content : "null") << endl;
}


//内存类；
class Memory {
	private:
		char* itsString;
		int itsLen;
	public:
		Memory();
		Memory(const char* content);
		~Memory();
		void SetMcontent(const char* content);
		char* GetMcontent() const {
			return this->itsString;
		}
};

Memory::Memory():itsString(NULL),itsLen(0) {
}

Memory::Memory(const char* content) {
	this->itsLen = strlen(content);
	this->itsString = new char[itsLen + 1];
	strcpy(itsString, content);
}

Memory::~Memory() {
	delete[] itsString;
}

void Memory::SetMcontent(const char* content) {
	if (!content) {
		delete[] itsString;
	}

	this->itsLen = strlen(content);
	this->itsString = new char[itsLen + 1];
	strcpy(itsString, content);
}



class Ch_mannager {
	public:
		Ch_mannager() {}
		~Ch_mannager() {}
		void run(int state);
		int sort(Device []);
		void memoryToDevice(Memory [],Device []);
};

void Ch_mannager::memoryToDevice(Memory m[],Device d[]) {
	int i;
	int maxDnum = 0;
	int flag = 0;
	static int time = 0;
	for(i = 0; i < 4; i++) {
		d[i].SetRequireState(1);
	}

	while(true) {
		cout << "-------------------------" << endl;
		cout << endl;
		cout << "time:" << time << endl;
		cout << endl;
//		if(d[0].GetContent())
//			cout << strlen( d[0].GetContent() )<< endl;

		if((d[0].GetContent() != NULL) && (strlen( d[0].GetContent()) == 4 ) ) {
			d[0].SetRequireState(0);
		}

		if((d[1].GetContent() != NULL) &&strlen(d[1].GetContent()) == 7) {
			d[1].SetRequireState(0);
		}
		if((d[2].GetContent() != NULL) &&strlen(d[2].GetContent()) == 6) {
			d[2].SetRequireState(0);
		}
		if((d[3].GetContent() != NULL) &&strlen(d[3].GetContent()) == 5) {
			d[3].SetRequireState(0);
		}
		for (i = 0; i<4; i++) {
			cout << "d[" << i << "]请求状态：" << d[i].GetRequireState() << endl ;
		}
		cout << endl;
		
		maxDnum = sort(d);

		d[maxDnum].SetContent( m[maxDnum].GetMcontent());
		d[maxDnum].SetRequireState(0);

		d[0].PrintDevice();
		d[1].PrintDevice();
		d[2].PrintDevice();
		d[3].PrintDevice();
		time+=5;
		for(i=0; i<4; i++) {

			if( ( time % d[i].GetRequireTime() ) == 0) {
				d[i].SetRequireState(1);
			}
		}
		
		cout << endl;
		
		
		if( ( strlen( d[0].GetContent() ) == 4&&strlen(d[1].GetContent()) == 7&&strlen(d[2].GetContent()) == 6&&strlen(d[3].GetContent()) == 5) )
			break;		
	}
}


int Ch_mannager::sort(Device d[]) {
	int index = 0;
	int i = 0;
	int max;
	max = 0;
	for(i = 0; i < 4; i++) {

		if(( d[i].GetRequireState() == 1)&&(d[i].Getpriority() > max)) {
			index = i;
			// cout << index <<"index" << endl;

			max = d[i].Getpriority();
			// cout << max << endl;
		}
	}
	cout << "device" << index << "获得服务" << endl;
	cout << endl;
	return index;
}


void Ch_mannager::run(int state) {
	while(true) {
		if(state == NONE) {
			cout<<"The cpu is doing some thing..."<<endl;
			cout<<"The cpu is doing some thing..."<<endl;
			break;
		}
		if(state == INIT) {
			cout<<"CPU is interrupted"<<endl;
			cout<<"This is a I/0 Init instruction,The channalManager is init thedevice..."<<endl;
			break;
		}
		if(state == FINISH) {
			cout<<"CPU is interrupted"<<endl;
			cout<<"This is a I/0 Finish instruction,The channalManager is close thedevice..."<<endl;
			break;
		}
	}
}

