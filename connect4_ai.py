import argparse, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROWS, COLS = 6, 7
WIN_LEN = 4

def c4_new():
    return np.zeros((2, ROWS, COLS), dtype=np.float32), 1

def c4_legal(board):
    return [c for c in range(COLS) if board[0,0,c]==0 and board[1,0,c]==0]

def c4_apply(board, player, action):
    b = board.copy()
    for r in range(ROWS-1,-1,-1):
        if b[0,r,action]==0 and b[1,r,action]==0:
            b[0 if player==1 else 1, r, action]=1
            break
    return b, -player

def c4_lines():
    lines=[]
    for r in range(ROWS):
        for c in range(COLS-WIN_LEN+1):
            lines.append([(r,c+i) for i in range(WIN_LEN)])
    for c in range(COLS):
        for r in range(ROWS-WIN_LEN+1):
            lines.append([(r+i,c) for i in range(WIN_LEN)])
    for r in range(ROWS-WIN_LEN+1):
        for c in range(COLS-WIN_LEN+1):
            lines.append([(r+i,c+i) for i in range(WIN_LEN)])
    for r in range(WIN_LEN-1,ROWS):
        for c in range(COLS-WIN_LEN+1):
            lines.append([(r-i,c+i) for i in range(WIN_LEN)])
    return lines

LINES = c4_lines()

def c4_winner(board):
    g = np.zeros((ROWS, COLS), dtype=np.int8)
    for p in [1,-1]:
        mask = board[0] if p==1 else board[1]
        for r in range(ROWS):
            for c in range(COLS):
                if mask[r,c]==1:
                    g[r,c]=p
    for line in LINES:
        s=sum(g[r,c] for r,c in line)
        if s==WIN_LEN: return 1
        if s==-WIN_LEN: return -1
    if all(board[0,0,c]!=0 or board[1,0,c]!=0 for c in range(COLS)): return 2
    return 0

def c4_obs(board, player):
    if player==1:
        return board.copy()
    return board[::-1].copy()

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1=nn.Conv2d(ch,ch,3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(ch)
        self.c2=nn.Conv2d(ch,ch,3,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(ch)
    def forward(self,x):
        y=F.relu(self.bn1(self.c1(x)))
        y=self.bn2(self.c2(y))
        return F.relu(x+y)

class Net(nn.Module):
    def __init__(self, ch=64, blocks=4):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(2,ch,3,padding=1,bias=False), nn.BatchNorm2d(ch), nn.ReLU())
        self.res=nn.Sequential(*[ResidualBlock(ch) for _ in range(blocks)])
        self.p_head=nn.Sequential(nn.Conv2d(ch,2,1), nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(), nn.Linear(2*ROWS*COLS, COLS))
        self.v_head=nn.Sequential(nn.Conv2d(ch,1,1), nn.BatchNorm2d(1), nn.ReLU(), nn.Flatten(), nn.Linear(ROWS*COLS, ch), nn.ReLU(), nn.Linear(ch,1), nn.Tanh())
    def forward(self,x):
        x=self.res(self.stem(x))
        p=self.p_head(x)
        v=self.v_head(x)
        return p,v

class MCTS:
    def __init__(self, net, sims=200, c_puct=1.5, dir_epsilon=0.25, dir_alpha=0.3, device='cpu'):
        self.net=net
        self.sims=sims
        self.c_puct=c_puct
        self.dir_epsilon=dir_epsilon
        self.dir_alpha=dir_alpha
        self.Q={}
        self.N={}
        self.P={}
        self.children={}
        self.term={}
        self.device=device
    def key(self, board, player):
        return (player, board.tobytes())
    def expand(self, board, player):
        k=self.key(board,player)
        if k in self.P: return
        obs=torch.tensor(c4_obs(board,player)).unsqueeze(0).to(self.device)
        p,v=self.net(obs)
        p=F.log_softmax(p,dim=1).exp().detach().cpu().numpy()[0]
        legal=c4_legal(board)
        mask=np.zeros(COLS,dtype=np.float32)
        for a in legal: mask[a]=1
        if mask.sum()==0: mask=np.ones(COLS,dtype=np.float32)
        p=p*mask
        if p.sum()==0: p=mask/mask.sum()
        p=p/p.sum()
        self.P[k]=p
        self.N[k]=np.zeros(COLS,dtype=np.int32)
        self.Q[k]=np.zeros(COLS,dtype=np.float32)
        self.children[k]=legal
        self.term[k]=c4_winner(board)
    def add_dirichlet(self, k):
        if random.random()<1:
            alpha=self.dir_alpha
            noise=np.random.dirichlet([alpha]*COLS)
            self.P[k]=(1-self.dir_epsilon)*self.P[k]+self.dir_epsilon*noise
    def simulate(self, board, player):
        k=self.key(board,player)
        self.expand(board,player)
        if self.term[k]==1: return 1
        if self.term[k]==-1: return -1
        if self.term[k]==2: return 0
        if np.sum(self.N[k])==0: self.add_dirichlet(k)
        legal=self.children[k]
        total=np.sum(self.N[k][legal])
        best_a=None
        best=-1e9
        for a in legal:
            q=self.Q[k][a]
            u=self.c_puct*self.P[k][a]*math.sqrt(total+1)/(1+self.N[k][a])
            s=q+u
            if s>best:
                best=s
                best_a=a
        a=best_a
        nb, np_player = c4_apply(board, player, a)
        v=self.simulate(nb, np_player)
        v=-v
        n=self.N[k][a]+1
        self.N[k][a]=n
        self.Q[k][a]=self.Q[k][a]+(v-self.Q[k][a])/n
        return v
    def policy(self, board, player, temperature=1.0):
        for _ in range(self.sims):
            self.simulate(board, player)
        k=self.key(board,player)
        counts=self.N[k].astype(np.float32)
        counts = counts if temperature>0 else np.where(counts==counts.max(),1,0).astype(np.float32)
        if temperature!=1 and temperature>0:
            counts = counts**(1/temperature)
        if counts.sum()==0:
            legal=c4_legal(board)
            if len(legal)==0:
                return np.ones(COLS,dtype=np.float32)/COLS
            p=np.zeros(COLS,dtype=np.float32)
            p[random.choice(legal)]=1.0
            return p
        return counts/counts.sum()

class Replay:
    def __init__(self, cap=50000):
        self.s=[]
        self.p=[]
        self.z=[]
        self.cap=cap
    def add(self, s, p, z):
        self.s.extend(s)
        self.p.extend(p)
        self.z.extend(z)
        if len(self.s)>self.cap:
            self.s=self.s[-self.cap:]
            self.p=self.p[-self.cap:]
            self.z=self.z[-self.cap:]
    def sample(self, bs):
        idx=np.random.randint(0,len(self.s),size=bs)
        x=torch.tensor(np.stack([self.s[i] for i in idx]))
        y1=torch.tensor(np.stack([self.p[i] for i in idx]))
        y2=torch.tensor(np.array([self.z[i] for i in idx], dtype=np.float32)).unsqueeze(1)
        return x,y1,y2
    def __len__(self):
        return len(self.s)

def self_play(net, games=10, sims=100, device='cpu'):
    data=[]
    for _ in range(games):
        board, player = c4_new()
        mcts=MCTS(net, sims=sims, device=device)
        traj_s=[]
        traj_p=[]
        players=[]
        t=1.0
        move_count=0
        while True:
            pi=mcts.policy(board, player, temperature=t)
            traj_s.append(c4_obs(board, player))
            traj_p.append(pi)
            a=int(np.random.choice(np.arange(COLS), p=pi))
            board, player = c4_apply(board, player, a)
            players.append(-player)
            move_count+=1
            t=1.0 if move_count<8 else 0.0
            w=c4_winner(board)
            if w!=0:
                z=0 if w==2 else w
                for i,pl in enumerate(players):
                    data.append((traj_s[i], traj_p[i], z*pl))
                break
    return data

def train_step(net, opt, batch):
    x,y_pi,y_v=batch
    x=x.to(next(net.parameters()).device)
    y_pi=y_pi.to(x.device)
    y_v=y_v.to(x.device)
    p,v=net(x)
    loss_p=F.cross_entropy(p, torch.argmax(y_pi,dim=1))
    loss_v=F.mse_loss(v, y_v)
    loss=loss_p+loss_v
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
    opt.step()
    return float(loss.item()), float(loss_p.item()), float(loss_v.item())

def evaluate(net_a, net_b, games=20, sims=100, device='cpu'):
    wa=0
    for g in range(games):
        board, player = c4_new()
        mcts_a=MCTS(net_a, sims=sims, device=device)
        mcts_b=MCTS(net_b, sims=sims, device=device)
        while True:
            if player==1:
                pi=mcts_a.policy(board, player, temperature=0.0)
            else:
                pi=mcts_b.policy(board, player, temperature=0.0)
            a=int(np.random.choice(np.arange(COLS), p=pi))
            board, player = c4_apply(board, player, a)
            w=c4_winner(board)
            if w!=0:
                if w==1: wa+=1
                elif w==-1: wa+=0
                break
    return wa, games-wa

def save_model(net, path):
    torch.save(net.state_dict(), path)

def load_model(net, path, map_location='cpu'):
    sd=torch.load(path, map_location=map_location)
    net.load_state_dict(sd)

def play_cli(model_path=None, device='cpu', sims=200):
    net=Net().to(device)
    if model_path and os.path.exists(model_path):
        load_model(net, model_path, map_location=device)
    board, player = c4_new()
    while True:
        print(render(board))
        if player==1:
            mcts=MCTS(net, sims=sims, device=device)
            pi=mcts.policy(board, player, temperature=0.0)
            a=int(np.random.choice(np.arange(COLS), p=pi))
            print("AI plays:", a)
        else:
            legal=c4_legal(board)
            print("Your move. Legal:", legal)
            a=None
            while a not in legal:
                try:
                    a=int(input("Column 0-6: ").strip())
                except:
                    a=None
        board, player = c4_apply(board, player, a)
        w=c4_winner(board)
        if w!=0:
            print(render(board))
            if w==1: print("AI wins")
            elif w==-1: print("You win")
            else: print("Draw")
            break

def render(board):
    g=np.zeros((ROWS,COLS),dtype=int)
    for r in range(ROWS):
        for c in range(COLS):
            if board[0,r,c]==1: g[r,c]=1
            elif board[1,r,c]==1: g[r,c]=-1
    s=""
    for r in range(ROWS):
        row=[]
        for c in range(COLS):
            row.append("X" if g[r,c]==1 else "O" if g[r,c]==-1 else ".")
        s+=" ".join(row)+"\n"
    s+="0 1 2 3 4 5 6"
    return s

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--selfplay_games", type=int, default=50)
    parser.add_argument("--sims", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_steps", type=int, default=200)
    parser.add_argument("--eval_games", type=int, default=20)
    parser.add_argument("--model_out", default="c4_az.pt")
    parser.add_argument("--model_in", default=None)
    parser.add_argument("--play", action="store_true")
    args=parser.parse_args()
    if args.play:
        play_cli(model_path=args.model_in, device=args.device, sims=args.sims)
        return
    device=args.device
    net=Net().to(device)
    if args.model_in and os.path.exists(args.model_in):
        load_model(net, args.model_in, map_location=device)
    opt=torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    rb=Replay()
    best=Net().to(device)
    best.load_state_dict(net.state_dict())
    for ep in range(1, args.epochs+1):
        data=self_play(best, games=args.selfplay_games, sims=args.sims, device=device)
        if len(data)==0: continue
        s=[d[0] for d in data]
        p=[d[1] for d in data]
        z=[d[2] for d in data]
        rb.add(s,p,z)
        for step in range(args.train_steps):
            if len(rb)<args.batch_size: break
            batch=rb.sample(args.batch_size)
            train_step(net, opt, batch)
        wa, wb = evaluate(net, best, games=args.eval_games, sims=args.sims//2, device=device)
        if wa>wb:
            best.load_state_dict(net.state_dict())
            save_model(best, args.model_out)
    save_model(best, args.model_out)

if __name__=="__main__":
    main()
