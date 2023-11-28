#include<iostream>
#include<cmath>
#include<cstring>

using namespace std;

const int N = 8; // huffman������󳤶� 

// huffman���ڵ���
class huff_p {
public:
    huff_p* r_child;  // ����ʵĽڵ�
    huff_p* l_child;  // С���ʵĽڵ�
    char op_mask[3];  // ָ����
    float p;          // ָ��ʹ�ø���
};

// ��С���ʽڵ�����ڵ���
class f_min_p {
public:
    f_min_p* next;
    char op_mask[3];   // ָ����
    float p;           // ָ��ʹ�ø���
    huff_p* huf_p;      // ��Ӧ��huffman���ڵ�
};

// huffman����ڵ���
class huff_code {
public:
    huff_code* next;
    float p;
    char op_mask[3];
    char code[N];      // huffman ����
};

// ����ָ���ģ��
f_min_p* input_instruct_set();

// ����huffman��
huff_p* creat_huffman_tree(f_min_p* head);

// ����С���ʽڵ��������ҵ�������С�Ľڵ�
f_min_p* fin_min(f_min_p* h);

// ����С���ʽڵ�������ɾ��ָ���ڵ�
f_min_p* del_min(f_min_p* h, f_min_p* p);

// ���½ڵ㰴���ʲ��뵽��С���ʽڵ������к��ʵ�λ��
void insert_n(f_min_p* h, f_min_p* p);

// ������С���ʽڵ㴴��huffman���ڵ�
huff_p* creat_huffp(f_min_p* p);

// �ݹ�����huffman����
void r_find(huff_p* p1, char code[], int i, huff_code* h);

// ����huffman����
void creat_huffman_code(huff_p* h1, huff_code* h);

// ���huffman����
void output_huffman(huff_code* head);

// ����ָ����huffman�����ƽ�������ֳ�
void cal_sort_length(huff_code* head);

// ��ӡhuffman��
void print(huff_p* h1);

int main() {
    f_min_p* h, *h1;
    huff_p* root;
    huff_code* head, *pl;

    h = input_instruct_set();
    h1 = h;
    root = creat_huffman_tree(h1);

    head = new huff_code;
    head->next = NULL;
    creat_huffman_code(root, head);

    output_huffman(head);
    cal_sort_length(head);

    pl = head->next;
    while (pl) {
        delete head;
        head = pl;
        pl = pl->next;
    }
    
    system("pause");
    return 0;
}

// ����ָ���ģ��
f_min_p* input_instruct_set() {
    f_min_p* head;
    f_min_p* h;
    h = new f_min_p;
    h->next = NULL;
    h->huf_p = NULL;
    head = h;
    int n;
    cout << "������ָ����:";
    cin >> n;
    cout << "������ָ���ţ�";
    cin >> h->op_mask;
    cout << "������ָ���ʹ�ø��ʣ�";
    cin >> h->p;

    f_min_p* point;
    f_min_p* p1 = head;

    for (int i = 0; i < n - 1; i++) {
        point = new f_min_p;
        cout << "������ָ���ţ�";
        cin >> point->op_mask;
        point->op_mask[2] = '\0';
        cout << "������ָ���ʹ�ø��ʣ�";
        cin >> point->p;
        point->huf_p = NULL;
        point->next = p1->next;
        p1->next = point;
        p1 = point;
    }
    return head;
}

// ����huffman��
huff_p* creat_huffman_tree(f_min_p* h) {
    f_min_p* h1, *min1, *min2, *comb;
    huff_p* head, *rd, *ld, *parent;
    h1 = h;

    min1 = fin_min(h1);
    ld = creat_huffp(min1);
    h1 = del_min(h1, min1);

    if (h1->next) {
        min2 = fin_min(h1);
    }
    else {
        min2 = h1;
    }

    rd = creat_huffp(min2);
    comb = new f_min_p;
    comb->next = NULL;
    comb->p = rd->p + ld->p;
    comb->op_mask[0] = '\0';
    comb->op_mask[1] = '\0';

    parent = creat_huffp(comb);

    insert_n(h1, comb);
    if (h1->next != NULL) {
        h1 = del_min(h1, min2);
    }

    parent->l_child = ld;
    parent->r_child = rd;

    comb->huf_p = parent;

    head = parent;
    while (h1->next != NULL) {
        min1 = fin_min(h1);
        if (min1->huf_p == NULL) {
            ld = creat_huffp(min1);
        }
        else {
            ld = min1->huf_p;
        }
        h1 = del_min(h1, min1);

        if (h1->next) {
            min2 = fin_min(h1);
        }
        else {
            min2 = h1;
        }
        if (min2->huf_p == NULL) {
            rd = creat_huffp(min2);
        }
        else {
            rd = min2->huf_p;
        }
        comb = new f_min_p;
        comb->next = NULL;
        comb->p = rd->p + ld->p;
        comb->op_mask[0] = '\0';
        comb->op_mask[1] = '\0';
        parent = creat_huffp(comb);
        if (h1 != NULL) {
            insert_n(h1, comb);
        }

        if (h1->next != NULL) {
            h1 = del_min(h1, min2);
        }
        if (h1->next == NULL && ld->p < rd->p) {
            huff_p* tmp = ld;
            ld = rd;
            rd = tmp;
        }
        parent->l_child = ld;
        parent->r_child = rd;
        comb->huf_p = parent;
        head = parent;

        if (h1->next == NULL) {
            break;
        }
    }
    delete comb;
    return head;
}

// ����С���ʽڵ��������ҵ�������С�Ľڵ�
f_min_p* fin_min(f_min_p* h) {
    f_min_p* h1, *p1;
    h1 = h;
    p1 = h1;
    float min = h1->p;
    h1 = h1->next;
    while (h1) {
        if (min > (h1->p)) {
            min = h1->p;
            p1 = h1;
        }
        h1 = h1->next;
    }
    return p1;
}

// ����С���ʽڵ�������ɾ��ָ���ڵ�
f_min_p* del_min(f_min_p* h, f_min_p* p) {
    f_min_p* p1, *p2;
    p1 = h;
    p2 = h;
    if (h == p) {
        h = h->next;
        delete p;
    }
    else {
        while (p1->next != NULL) {
            p1 = p1->next;
            if (p1 == p) {
                p2->next = p1->next;
                delete p;
                break;
            }
            p2 = p1;
        }
    }
    return h;
}

// ���½ڵ㰴���ʲ��뵽��С���ʽڵ������к��ʵ�λ��
void insert_n(f_min_p* h, f_min_p* p1) {
    p1->next = h->next;
    h->next = p1;
}

// ������С���ʽڵ㴴��huffman���ڵ�
huff_p* creat_huffp(f_min_p* d) {
    huff_p* p1;
    p1 = new huff_p;
    p1->l_child = NULL;
    p1->r_child = NULL;
    p1->p = d->p;
    p1->op_mask[0] = d->op_mask[0];
    p1->op_mask[1] = d->op_mask[1];
    return p1;
}

// �ݹ�����huffman����
void r_find(huff_p* p1, char code[], int i, huff_code* h) {
    if (p1->l_child) {
        code[i] = '1';
        r_find(p1->l_child, code, i + 1, h);
    }
    if (p1->op_mask[0] != '\0') {
        huff_code* p2 = new huff_code;
        p2->op_mask[0] = p1->op_mask[0];
        p2->op_mask[1] = p1->op_mask[1];
        p1->op_mask[2] = '\0';
        p2->p = p1->p;
        int j = 0;
        for (; j < i; j++) {
            p2->code[j] = code[j];
        }
        p2->code[j] = '\0';
        p2->next = h->next;
        h->next = p2;
    }
    if (p1->r_child) {
        code[i] = '0';
        r_find(p1->r_child, code, i + 1, h);
    }
    delete p1;
}

// ����huffman����
void creat_huffman_code(huff_p* h1, huff_code* h) {
    int i = 0;
    char code[N] = { '\0' };
    r_find(h1, code, i, h);
}

// ���huffman����
void output_huffman(huff_code* head) {
    huff_code* h = head->next;
    cout << "ָ��\t" << "����\t" << "����" << endl;
    cout << "---------------------------------" << endl;
    while (h) {
        h->op_mask[2] = '\0';
        cout << h->op_mask << ":\t" << h->p << "\t" << h->code << endl;
        h = h->next;
    }
    cout << "---------------------------------" << endl;
    cout << endl;
}

// ����ָ����huffman�����ƽ�������ֳ�
void cal_sort_length(huff_code* head) {
    huff_code* h = head->next;
    double j = 0;
    float one_length = 0;
    float per_length = 0;
    float ext_length = 0;//��1-2-3-5��չ�������С����Ϊ��

    while (h) {
        float length = 0;
        int i = 0;
        while (h->code[i] != '\0') {
            length++;
            i++;
        }
        one_length = h->p * length;
        per_length = per_length + one_length;
        h = h->next;
        j++;
    }
    int i1 = int(j);
    huff_code* p2 = head->next;
    float* p_a = new float[i1];
    //sortָ�����
    int i0 = 0;
    while (p2) {
        p_a[i0++] = p2->p;
        p2 = p2->next;
    }
    float max, temp;
    int l;
    for (int s = 0; s < i1; s++) {
        max = p_a[s];
        l = s;
        for (int k = s + 1; k < i1; k++) {
            if (max < p_a[k]) {
                max = p_a[k];
                l = k;
            }
        }
        temp = p_a[s];
        p_a[s] = max;
        p_a[l] = temp;
    }
    //����1-2-3-5��չ��������ƽ�����볤��
    float* code_len = new float[i1];
    code_len[0] = 1;
    code_len[1] = 2;
    code_len[2] = 3;
    code_len[3] = 5;
    for (int i = 4; i < j; i++) {
        code_len[i] = 5;
    }
    l = 0;
    while (l < i1) {
        ext_length = ext_length + code_len[l] * p_a[l];
        l++;
    }

    //����ȳ�����ƽ�����ȣ�
    int q_length = ceil(log10(j) / log10(2));

    cout << "��ָ������� huffman �����ƽ������Ϊ��" << per_length << endl;
    cout << "�ȳ������ƽ������Ϊ��" << q_length << endl;
    cout << "��1-2-3-5����չ��������ƽ�����볤��Ϊ��" << ext_length;
    cout << endl;
    cout << endl;
    if (q_length > per_length) {
        cout << "�ɼ� huffman �����ƽ������Ҫ�ȵȳ������ƽ�����ȶ�" << endl;
    }
    else {
        cout << "huffman ��������������ϸ�鿴�㷨���Լ������ָ��ĸ���֮���Ƿ����1��" << endl;
    }
    if (ext_length > per_length) {
        cout << "�ɼ� huffman �����ƽ������Ҫ��1-2-3-5��չ��������ƽ�����ȶ�" << endl;
    }
    else {
        cout << "huffman ��������������ϸ�鿴�㷨���Լ������ָ��ĸ���֮���Ƿ����1��" << endl;
    }
}

