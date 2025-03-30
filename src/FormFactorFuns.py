import numpy as np

def GmapInHexgon(Lp):
    """
    Gmap, Indmap = GmapInHexgon(Lp) gives a dictionary that maps tuples (x, y), G' = x*G2 + y*G3
    to an index ranging from 1 to NL = 1 + 3*Lp + 3*Lp^2
    """
    # N_L = N_{L-1} + L*6, N_0 = 1, N_L = 1 + 3L + 3L^2
    NL = 1 + 3*Lp + 3*Lp^2
    G2 = np.array([np.sqrt(3)/2, 1/2])  # * 4 * np.pi / np.sqrt(3)
    G3 = np.array([-np.sqrt(3)/2, 1/2])  # * 4 * np.pi / np.sqrt(3)
    # normG = np.linalg.norm(G2)
    L = 2 * Lp
    k0 = np.array([0, -L * G2[1]])
    GpMapToInd = {}
    IndMapToGp = {}
    tempind = 1
    for indx in range(L + 1):
        for indy in range(L + 1):
            curpos = k0 + indx * G2 + indy * G3
            if np.sum(curpos**2) <= Lp**2 + 0.01:  # smaller than Lp |G|, 0.01 for calculation error
                tup = (indx - Lp, indy - Lp)
                # print(tup, np.linalg.norm(curpos))
                GpMapToInd[tup] = tempind
                IndMapToGp[tempind] = tup
                tempind += 1
    return GpMapToInd, IndMapToGp

def FullMomentumMap(Nx, Ny):
    KMapToInd = {}
    IndMapToK = {}
    for indx in range(1, Nx+1):
        for indy in range(1, Ny+1):
            tup = (indx, indy)
            ind = (indx-1)*Ny + indy
            KMapToInd[tup] = ind
            IndMapToK[ind] = tup
    return KMapToInd, IndMapToK

def WilsonLoopSingle(curInd, Nx, Ny, FormFactors, etaxy=False):
    NBZ = FormFactors.shape[1] // (Nx * Ny)
    Lp = int(round((np.sqrt(12*NBZ-3)-3)/6))
    GpmapLp, IndmapLp = GmapInHexgon(Lp)
    KMapToInd, IndMapToK = FullMomentumMap(Nx, Ny)
    kx0, ky0 = IndMapToK[curInd]

    def GetLoc(kx0, ky0, δkx, δky):
        kx, ky = kx0 + δkx, ky0 + δky
        BZlocx, BZlocy = 0, 0
        if kx > Nx:
            kx -= Nx
            BZlocx = 1
        if ky > Ny:
            ky -= Ny
            BZlocy = 1
        Kloc = [kx, ky]
        BZloc = [BZlocx, BZlocy]
        return Kloc, BZloc

    N = Nx * Ny
    KLoop = [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]
    IndLs = np.zeros(4, dtype=int)
    IndRs = np.zeros(4, dtype=int)
    for k in range(4):
        Kloc1, BZloc1 = GetLoc(kx0, ky0, KLoop[k][0], KLoop[k][1])
        Kloc2, BZloc2 = GetLoc(kx0, ky0, KLoop[k+1][0], KLoop[k+1][1])
        if BZloc1 == [0, 0]:
            IndLs[k] = KMapToInd[tuple(Kloc1)]
            BZind = GpmapLp[tuple(BZloc2)]
            IndRs[k] = N * (BZind-1) + KMapToInd[tuple(Kloc2)]
        elif BZloc2 == [0, 0]:
            IndRs[k] = KMapToInd[tuple(Kloc2)]
            BZind = GpmapLp[tuple(BZloc1)]
            IndLs[k] = N * (BZind-1) + KMapToInd[tuple(Kloc1)]
        else:
            BZloc2 = [bz2 - bz1 for bz2, bz1 in zip(BZloc2, BZloc1)]
            BZloc1 = [0, 0]
            IndLs[k] = KMapToInd[tuple(Kloc1)]
            BZind = GpmapLp[tuple(BZloc2)]
            IndRs[k] = N * (BZind-1) + KMapToInd[tuple(Kloc2)]
        IndLs[k] -= 1
        IndRs[k] -= 1 # for python
    if etaxy:
        res = 0.0
        for k in range(4):
            if IndLs[k] <= N:
                res += 1 - abs(FormFactors[IndLs[k], IndRs[k]])**2
            else:
                res += 1 - abs(FormFactors[IndRs[k], IndLs[k]].conj())**2
        return res

    res = 1 + 0j
    for k in range(4):
        #print(IndLs[k], IndRs[k])
        if IndLs[k] <= N:
            res *= FormFactors[IndLs[k], IndRs[k]]
        else:
            res *= FormFactors[IndRs[k], IndLs[k]].conj()
    return np.imag(np.log(res))

def WilsonLoopFull(Nx, Ny, FormFactors, etaxy=False):
    N = Nx * Ny
    res = np.zeros(N)
    for k in range(N):
        res[k] = WilsonLoopSingle(k+1, Nx, Ny, FormFactors, etaxy=etaxy)
    return res

def FSoverlapSingle(curInd, Nx, Ny, FormFactors):
    NBZ = FormFactors.shape[1] // (Nx * Ny)
    Lp = int(round((np.sqrt(12*NBZ-3)-3)/6))
    GpmapLp, IndmapLp = GmapInHexgon(Lp)
    KMapToInd, IndMapToK = FullMomentumMap(Nx, Ny)
    kx0, ky0 = IndMapToK[curInd]

    def GetLoc(kx0, ky0, δkx, δky):
        kx, ky = kx0 + δkx, ky0 + δky
        BZlocx, BZlocy = 0, 0
        if kx > Nx:
            kx -= Nx
            BZlocx = 1
        elif kx < 1:
            kx += Nx
            BZlocx = -1
        if ky > Ny:
            ky -= Ny
            BZlocy = 1
        elif ky < 1:
            ky += Ny
            BZlocy = -1
        Kloc = [kx, ky]
        BZloc = [BZlocx, BZlocy]
        return Kloc, BZloc

    N = Nx * Ny
    Knear = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    IndRs = np.zeros(4, dtype=int)
    for k in range(4):
        Kloc1, BZloc1 = GetLoc(kx0, ky0, Knear[k][0], Knear[k][1])
        BZind = GpmapLp[tuple(BZloc1)]
        IndRs[k] = N * (BZind-1) + KMapToInd[tuple(Kloc1)]

    etaxx = 0.0
    for k in range(2):
        etaxx += 1 - np.abs(FormFactors[curInd-1, IndRs[k]-1])**2

    etayy = 0.0
    for k in range(2, 4):
        etayy += 1 - np.abs(FormFactors[curInd-1, IndRs[k]-1])**2

    return (etaxx + etayy) / 2

def FSoverlapFull(Nx, Ny, FormFactors):
    N = Nx * Ny
    res = np.zeros(N)
    for k in range(N):
        res[k] = FSoverlapSingle(k+1, Nx, Ny, FormFactors)
    return res