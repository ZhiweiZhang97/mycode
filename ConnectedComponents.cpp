#include<iostream>
#include<vector>
#include<stack>
using namespace std;
static const int MAX = 100000;
static const int NIL = -1;

int n;
vector<int> G[MAX];
int color[MAX];

void dfs(int r, int c){
	stack<int> s;
	s.push(r);
	color[r] = c;
	while(!s.empty()){
		int u = s.top();
		s.pop();
		for(int i = 0;i<G[u].size();i++){
			int v = G[u][i];
			if(color[v] == NIL){
				color[v] = c;
				s.push(v);
			}
		}
	}
}
void assignColor(){
	int id = 1;
	for(int i = 0;i < n;i++) color[i] = NIL;
	for(int u = 0;u < n;u++){
		if(color[u] == NIL) dfs(u,id++);
	}
}
int main(){
	int s,t,m,q;
	int i;
	cin >> n >> m;
	for(i = 0;i < m;i++){
		cin >> s >> t;
		G[s].push_back(t);
		G[t].push_back(s);
	}
	assignColor();
	cin >> q;
	for(i = 0;i < q;i++){
		cin >> s >> t;
		if(color[s] == color[t]){
			cout << "yes" <<endl;
		}else{
			cout << "no" << endl;
		}
	}
	return 0;
}