import numpy as np
import torch

class torchMesh():
    def __init__(self, f_dir, inverse=True):
        if torch.cuda.is_available():
            print(' --- On GPU --- ')
            self.device = 'cuda'
        else:
            print(' --- On CPU --- ')
            self.device = 'cpu'

        p1, p2, p3, p4, p5 = None, None, None, None, None
        self.v, self.f = [], []
        with open(f_dir) as f:
            line = f.readline()
            print('#--- File Header: ', line)
            while line:
                line = f.readline()
                if line[:2]=='f ':
                    v_list = []
                    for v in line.split(' ')[1:]:
                        v_list.append(int(v.split('/')[0])-1)
                    v1, v2, v3, v4 = v_list
                    self.f.append([v1, v2, v3])
                    self.f.append([v1, v3, v4])
                    if p1 is None:
                        p1, p2, p3 = v1, v2, v3
                    elif (v2 in [p2, p3]) and (v3 in [p2, p3]): p4, p5 = v3, v4
                    elif (v1 in [p2, p3]) and (v2 in [p2, p3]): p4, p5 = v2, v3
                    elif (v3 in [p2, p3]) and (v4 in [p2, p3]): p4, p5 = v3, v2
                    elif (v1 in [p2, p3]) and (v4 in [p2, p3]): p4, p5 = v1, v2

                elif line[:2]=='v ':
                    v_x = [float(x) for x in line.split(' ')[1:]]
                    if inverse: self.v.append([v_x[0], -v_x[2], v_x[1],])
                    else: self.v.append([v_x[0], v_x[1], v_x[2],])

        self.v, self.f = np.array(self.v), np.array(self.f)
        print('#---- Loaded: ', f_dir, ' ----#')
        print('Vertices: ', self.v.shape, '; Triangle Faces: ', self.f.shape)
        self._gen_fv()
        self.axes = np.array([self.v[p3] - self.v[p2], self.v[p1] - self.v[p2], self.v[p5] - self.v[p4]]) # 3, 3
        self.axes = torch.from_numpy(self.axes).float().to(self.device)
        self.origin = torch.from_numpy(np.array([self.v[p2],])).float().to(self.device) # 1, 3
        self.center = self.origin + torch.sum(self.axes * 0.5, dim=0, keepdim=True)
        self.f = torch.from_numpy(self.f).int().to(self.device)
        self.v = torch.from_numpy(self.v).float().to(self.device)


    def _gen_fv(self):
        self.f_v = []
        for f in self.f:
            self.f_v.append([self.v[f[0]], self.v[f[1]],self.v[f[2]]])
        self.f_v = np.array(self.f_v)
        print('Face Vertices: ', self.f_v.shape) # f,p=3,3
        self.f_v = torch.from_numpy(self.f_v).float().to(self.device)

    def _intersect(self, rayo, rayd, eps=1e-8):
        # rayo: [n,3], rayd: [n,3]
        # return int_t>0 if intersect, else int_t=0

        rayn, fn = rayd.size(0), self.f_v.size(0)

        edge1 = self.f_v[:, 1] - self.f_v[:, 0]  # [f,3]
        edge2 = self.f_v[:, 2] - self.f_v[:, 0]  # [f,3]

        edge1= edge1[None, :, :].repeat(rayn, 1, 1) # [n,f,3]
        edge2= edge2[None, :, :].repeat(rayn, 1, 1) # [n,f,3]
        rayd_ = rayd[:, None, :].repeat(1, fn, 1) # [n,f,3]

        h = torch.cross(rayd_, edge2)
        a = torch.sum(edge1 * h, dim=-1)

        f = 1. / (a + eps)
        s = rayo[:, None, :] - self.f_v[None, :, 0]
        u = f * torch.sum(s * h, dim=-1)

        q = torch.cross(s, edge1)
        v = f * torch.sum(rayd_ * q, dim=-1)

        t = f * torch.sum(edge2 * q, dim=-1)  # n, f

        non_cond = (a > -eps) & (a < eps)
        non_cond = non_cond | ((u < 0) | (u > 1))
        non_cond = non_cond | ((v < 0) | (u + v > 1))
        non_cond = non_cond | (t < eps)

        max_t, _ = torch.max(t, dim=-1, keepdim=True)  # n,1
        int_t = torch.where(non_cond, max_t + 1, t)
        int_t, t_ind = torch.min(int_t, dim=-1, keepdim=True)
        cond = (max_t + 1 - int_t) > 0

        zeros_t = torch.zeros_like(int_t).to(self.device)
        int_t = torch.where(cond, int_t, zeros_t)  # n,1

        return int_t, t_ind, cond

    def intersect(self, rayo, rayd, bs=10000):

        n = rayo.size(0)
        n_iter = n//bs if n%bs==0 else n//bs+1
        rayd = torch.nn.functional.normalize(rayd, p=2, dim=-1)

        int_ts, t_inds, conds = [], [], []
        for i in range(n_iter):
            st, ed = i*bs, min(n, (i+1)*bs)
            t_chunk, ind_chunk, cond_chunk = self._intersect(rayo[st:ed], rayd[st:ed])
            int_ts.append(t_chunk)
            t_inds.append(ind_chunk)
            conds.append(cond_chunk)
        int_t = torch.cat(int_ts, dim=0)
        t_ind = torch.cat(t_inds, dim=0)
        cond = torch.cat(conds, dim=0)
        int_p = rayo + int_t * rayd

        zeros_p = torch.zeros_like(int_p).to(self.device)
        int_p = torch.where(cond, int_p, zeros_p)

        return int_p, int_t, t_ind, cond

    def _face_sample(self, n_f):

        fn = self.f_v.size(0)

        edge1 = self.f_v[:, 1] - self.f_v[:, 0]  # [f,3]
        edge2 = self.f_v[:, 2] - self.f_v[:, 0]  # [f,3]

        rand_t = torch.rand(1, fn, n_f, 2).to(self.device)
        rand_t = torch.cat([rand_t, 1. - rand_t], dim=0)

        t_r = torch.sum(torch.square(rand_t), dim=-1)
        _, sel_ind = torch.min(t_r, dim=0)
        sel_ind = sel_ind[..., None].repeat(1, 1, 2)

        rand_t = torch.gather(rand_t, 0, sel_ind[None, ...])
        rand_t = torch.reshape(rand_t, (fn, n_f, 2))

        edge1_, edge2_ = edge1[:, None, :], edge2[:, None, :]
        sampled_p = self.f_v[:, None, 0] + edge1_ * rand_t[:, :, :1] + edge2_ * rand_t[:, :, 1:]

        return sampled_p # f,n_f,3

    def _safe_norm(self, x, axis=-1, eps=1e-7):
        r = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
        r = np.where(r > 0, r, eps)
        return x/r

    def _cal_r(self, x, dim=-1):
        return torch.sqrt(torch.sum(torch.square(x), dim=dim, keepdim=True))



