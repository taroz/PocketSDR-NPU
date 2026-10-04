#!/usr/bin/env python3
#  compare tracking quality of bench_sdr_trk outputs at different sampling rates
#  usage: ana_trk_dec.py ref_16MHz.txt name=test.txt=fs_MHz ...
import sys,collections,statistics as st
C=299792458.0
def rd(f,fs):
    d=collections.defaultdict(list)
    for l in open(f):
        v=l.split(); d[int(v[1])].append((int(v[2])/fs,float(v[4]),float(v[5]),float(v[8])))
    return d
def stat(L):
    if len(L)<2500: return None
    L=L[1000:]
    cn0=st.mean(x[3] for x in L)
    co=[x[1] for x in L]; W=100
    res=[co[i]-st.mean(co[i-W//2:i+W//2]) for i in range(W//2,len(co)-W//2,5)]
    cn=st.pstdev(res)*C
    fd=[x[2] for x in L]; fdj=st.pstdev([fd[i+1]-fd[i] for i in range(len(fd)-1)])/2**0.5
    return cn0,cn,fdj
ref=rd(sys.argv[1],16e6); runs=[(a.split('=')[0],rd(a.split('=')[1],float(a.split('=')[2])*1e6)) for a in sys.argv[2:]]
print('%-8s'%'PRN'+''.join('%-26s'%('16MHz' if i==0 else runs[i-1][0]) for i in range(len(runs)+1)))
print('%-8s'%''+'cn0  code(m) fdjit(Hz)    '*(len(runs)+1))
agg=collections.defaultdict(list)
for p in sorted(ref):
    s0=stat(ref[p])
    if not s0: continue
    row='%-8d'%p+'%4.1f %7.3f %7.3f      '%s0
    for name,d in runs:
        s=stat(d.get(p,[]))
        if s: row+='%4.1f %7.3f %7.3f      '%s; agg[name].append((s[0]-s0[0],s[1]/s0[1],s[2]/s0[2]))
        else: row+='%-26s'%'lost'
    print(row)
for name,_ in runs:
    v=agg[name]
    print('%-8s mean dCN0=%+.2f dB  code noise x%.2f  fd jitter x%.2f (n=%d)'%(name,st.mean(x[0] for x in v),st.mean(x[1] for x in v),st.mean(x[2] for x in v),len(v)))
