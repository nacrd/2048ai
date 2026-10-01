// MIT; move and evaluation semantics derived from the upstream 2048 rules.
typedef unsigned char u8;
__device__ int index_for(int n, int d, int line, int i) {
  if(d==0) return i*n+line;
  if(d==1) return line*n+n-1-i;
  if(d==2) return (n-1-i)*n+line;
  return line*n+i;
}
__device__ bool move_board(const u8* b,u8* out,int n,int d,double* score) {
  for(int i=0;i<n*n;i++) out[i]=0;
  *score=0;
  for(int line=0;line<n;line++) {
    u8 vals[6]; int count=0;
    for(int i=0;i<n;i++){ u8 v=b[index_for(n,d,line,i)]; if(v) vals[count++]=v; }
    int slot=0;
    for(int k=0;k<count;k++) {
      int v=vals[k]; if(k+1<count && vals[k+1]==v) { v++; *score+=exp2((double)v); k++; }
      out[index_for(n,d,line,slot++)]=(u8)v;
    }
  }
  for(int i=0;i<n*n;i++) if(b[i]!=out[i]) return true;
  return false;
}
__device__ double evaluate(const u8* b,int n) {
  int empty=0,smooth=0,merges=0,monotonic=0,maximum=0;
  for(int r=0;r<n;r++) for(int c=0;c<n;c++) {
    int v=b[r*n+c]; if(!v) empty++; maximum=max(maximum,v);
    if(c+1<n) { int w=b[r*n+c+1]; if(v&&w) {smooth+=abs(v-w); if(v==w)merges+=v;} }
    if(r+1<n) { int w=b[(r+1)*n+c]; if(v&&w) {smooth+=abs(v-w); if(v==w)merges+=v;} }
  }
  for(int axis=0;axis<2;axis++) for(int line=0;line<n;line++) {
    int up=0,down=0;
    for(int i=0;i<n-1;i++){int a=b[axis?i*n+line:line*n+i],v=b[axis?(i+1)*n+line:line*n+i+1];a*=a;v*=v;up+=max(0,v-a);down+=max(0,a-v);}
    monotonic+=min(up,down);
  }
  bool corner=max(max((int)b[0],(int)b[n-1]),max((int)b[n*(n-1)],(int)b[n*n-1]))==maximum;
  if(!empty&&!merges)return -1000000000.0;
  int snake=-2147483647;
  for(int orientation=0;orientation<8;orientation++){
    int value=0,previous=0;
    for(int rank=0;rank<n*n;rank++){
      int r=rank/n,c=rank%n;if(r%2)c=n-1-c;
      if(orientation&1){int t=r;r=c;c=t;}if(orientation&2)r=n-1-r;if(orientation&4)c=n-1-c;
      int v=b[r*n+c];value+=v*v*(n*n-rank)*4;if(rank){int d=max(0,v-previous);value-=d*d*35;}previous=v;
    }
    snake=max(snake,value);
  }
  return empty*(280+maximum*20)-smooth*8+merges*35-monotonic*12+maximum*maximum*20+(corner?maximum*maximum*35:0)+snake;
}
__device__ unsigned next_uint(unsigned& x){ x^=x<<13;x^=x>>17;x^=x<<5;return x; }
__device__ void spawn(u8* b,int n,unsigned& rng){
  int available[36],count=0;for(int x=0;x<n;x++)for(int y=0;y<n;y++){int i=y*n+x;if(!b[i])available[count++]=i;}
  if(count){int v=(double)next_uint(rng)/4294967296.0<0.9?1:2;int index=(int)((double)next_uint(rng)/4294967296.0*count);b[available[index]]=(u8)v;}
}
__device__ unsigned trajectory_seed(unsigned seed,int direction,unsigned ident){
  unsigned x=seed^((unsigned)(direction+1)*0x9e3779b9u)^((ident+1)*0x85ebca6bu);x^=x>>16;x*=0x7feb352du;x^=x>>15;return x?x:0x6d2b79f5u;
}
__device__ int greedy(const u8* b,int n){
  int best=-1;double value=-1e100;u8 out[36];
  for(int d=0;d<4;d++){double score;if(move_board(b,out,n,d,&score)){double q=evaluate(out,n)+log2(score+1)*12;if(q>value){best=d;value=q;}}}
  return best;
}
extern "C" __global__ void rollout(const u8* input,int n,const int* directions,int numDirections,int offset,int count,unsigned seed,unsigned long long horizon,int objective,int target,double* results){
  int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=numDirections*count)return;
  int dirIndex=j/count,ident=j%count,action=directions[dirIndex];unsigned rng=trajectory_seed(seed,action,offset+ident);
  u8 b[36],out[36];for(int i=0;i<n*n;i++)b[i]=input[i];double score=0;
  for(unsigned long long step=0;step<horizon;step++){
    if(objective){bool won=false;for(int i=0;i<n*n;i++)if(b[i]>=target)won=true;if(won){results[j]=1;return;}}
    if(action<0)break;double delta;if(!move_board(b,out,n,action,&delta))break;score+=delta;
    for(int i=0;i<n*n;i++)b[i]=out[i];spawn(b,n,rng);action=greedy(b,n);
  }
  if(objective){bool won=false;for(int i=0;i<n*n;i++)if(b[i]>=target)won=true;results[j]=won?1:0;}else results[j]=score;
}
extern "C" __global__ void moves(const u8* input,int n,int count,u8* results,double* scores,int* changed){
  int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=count*4)return;int d=j%4,board=j/4;
  changed[j]=move_board(input+board*n*n,results+j*n*n,n,d,scores+j);
}
