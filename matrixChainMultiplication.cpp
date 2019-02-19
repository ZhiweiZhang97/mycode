#include<iostream>
#include<algorithm>
using namespace std;
#define min(a,b) ((a)<(b))?(a):(b)

static const int N = 100;
int main(){
	int n, p[N+1], m[N+1][N+1];
	int i, j, k, l;
	cin >> n;
	for (i = 1;i <= n;i++){
		cin >> p[i-1] >> p[i];
	}
	for(i = 1;i <= n;i++) m[i][i] = 0;
	for(l = 2;l <= n;l++){
		for(i = 1;i <= n - l + 1;i++){
			j = i + l - 1;
			m[i][j] = (1<<21);
			for(k = i;k <= j - 1;k++){
				m[i][j] = min(m[i][j],m[i][k] + m[k+1][j] + p[i-1] * p[k] * p[j]);
			}
		}
	}
	cout << m[1][n] <<endl;
	return 0;
}