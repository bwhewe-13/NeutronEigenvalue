import numpy as np
import matplotlib.pyplot as plt

def color_negative_red(val):
    color = 'red' if val == 0 else 'black'
    return 'color: %s' % color

def create_grid(R,I):
    """Create the cell edges and centers for a domain of size R and I cells
    Args:   R: size of domain
            I: number of cells
    Returns:    Delt_r: the width of each cell
                centers: the cell centers of the grid
                edges: the cell edges of the grid
                """
    
    Delta_r = float(R)/I #calculate distance between cells
    centers = np.arange(I)*Delta_r + 0.5*Delta_r #list of cell center locations within the range, len equal to I
    edges = np.arange(I+1)*Delta_r #list of cell edges within the domain size, len equal to I+1
    return Delta_r, centers, edges

def swap_rows(A, a, b):
    '''Swap two rows in a matrix: switch row a with row b
    Args:   A: matrix to perform row swaps on
            a: row index of matrix
            b: row index of matrix
    Returns: Nothing
    Side Effects: Changes A to have rows a and b swapped
    '''
    assert(a>=0) and (b>=0)
    N = A.shape[0] #number of rows
    assert (a<N) and (b<N) #less than because 0-based indexing
    temp = A[a,:].copy()
    A[a,:] = A[b,:].copy()
    A[b,:] = temp.copy()
    
def LU_factor(A,LOUD=True):
    '''Factor in place A in L*U=A. The lower triangular parts of A are 
    the L matrix. The L has implied ones on the diagonal.
    Args: 
        A: N by N array
    Returns: 
        a vector holding the order of the rows,
        relative to the original order
    Side Effects: 
        A is factored in place
    '''
    [Nrow,Ncol]= A.shape
    assert Nrow==Ncol
    N = Nrow
    #create scale factors
    s = np.zeros(N)
    count = 0
    row_order = np.arange(N)
    for row in A:
        s[count] = np.max(np.fabs(row))
        count += 1
    if LOUD:
        print('s = ',s)
    if LOUD:
        print('Original Matrix is \n',A)
    for column in range(0,N):
        #swap rows if needed
        largest_pos = np.argmax(np.fabs(A[column:N,column]/s[column]))+column
        if (largest_pos != column):
            if (LOUD):
                print('Swapping row',column,'with row',largest_pos)
                print('Pre swap\n',A)
            swap_rows(A,column,largest_pos)
            #keep track of changes to RHS
            tmp = row_order[column]
            row_order[column] = row_order[largest_pos]
            row_order[largest_pos] = tmp
            #re-order s
            tmp = s[column]
            s[column] = s[largest_pos]
            s[largest_pos] = tmp
            if (LOUD):
                print('A =\n',A)
        for row in range(column+1,N):
            mod_row = A[row]
            factor = mod_row[column]/A[column,column]
            mod_row = mod_row - factor*A[column,:]
            # put the factor in the correct place in the modified row
            mod_row[column] = factor
            # only take the part of the modified row we need
            mod_row = mod_row[column:N]
            A[row,column:N] = mod_row
    return row_order


def LU_solve(A,b,row_order):
    '''Take a LU factorized matrix and solve it for RHS b
    Args:
        A: N by N array that has been LU factored with 
        assumed 1's on the diagonal of the L matrix
        b: N by 1 array of righthand side
        row_order: list giving the re-ordered equations
        from the LU factorization with pivoting
    Returns:
        x: N by 1 array of solutions
    '''
    [Nrow,Ncol] = A.shape
    assert Nrow == Ncol
    assert b.size == Ncol
    assert row_order.max() == Ncol-1
    N = Nrow
    # Reorder the equations
    tmp = b.copy()
    for row in range(N):
        b[row_order[row]] = tmp [row]
        
    x = np.zeros(N)
    # temporary vector for L^-1 b
    y = np.zeros(N)
    # forward solve
    for row in range(N):
        RHS = b[row]
        for column in range(0,row):
            RHS -= y[column]*A[row,column]
        y[row] = RHS
    # back solve
    for row in range(N-1,-1,-1):
        RHS = y[row]
        for column in range(row+1,N):
            RHS -= x[column]*A[row,column]
        x[row] = RHS/A[row,row]
    return x

def bisection(f,a,b,epsilon=1.0e-6):
    ''' Fine the root of the function f via bisection
    where the root lies within [a,b]
    Inputs: 
        f: function to find root of
        a: left-side of interval
        b: right-side of interval
        epsilon: tolerance
    Returns:
        estimate of root
    '''
    assert(b>a)
    fa = f(a)
    fb = f(b)
#     assert (fa*fb < 0)
    delta = b-a
    print('We expect',int(np.ceil(np.log(delta/epsilon)/np.log(2))),'iterations')
    iterations = 0
    while (delta > epsilon):
        c = (a+b)*0.5
        fc = f(c)
        if (fa*fc < 0):
            b = c
            fb = fc
        elif (fb*fc < 0):
            a = c
            fa = fc
        else:
            return c
        delta = b-a
        iterations += 1
    print('It took',iterations,'iterations')
    return c #return midpoint of interval 

def ridder(f,a,b,epsilon=1.0e-6):
    '''Find the froot of the function f via Ridder's Method
    where the root lies within [a,b]
    Args:
        f: function to find root of
        a: left-side of interval
        b: right-side of interval
        epsilon: tolerance
    Returns:
        estimate of root
    '''
    assert(b>a)
    fa = f(a)
    fb = f(b)
    assert (fa*fb < 0)
    delta = b * a
    iterations = 0
    residual = 1.0
    while (np.fabs(residual) > epsilon):
        c = 0.5*(b+a)
        d = 0.0
        fc = f(c)
        if (fa - fb > 0):
            d = c + (c-a)*fc/np.sqrt(fc**2-fa*fb)
        else:
            d = c - (c-a)*fc/np.sqrt(fc**2-fa*fb)
        fd = f(d)
        # now see which part of interval root is in
        if (fa*fd < 0):
            b = d
            fb = fd
        elif (fb*fd < 0):
            a = d
            fa = fd
        residual = fd
        iterations += 1
#         print('temp value',d)
    print('It took',iterations,'iterations')
    return d #return c
    
def get_pos_from_i_g(i,g,I,G): #change the cell position
    return g*I+i
def get_g_i_from_pos(pos,I,G): #change the group number 
    return pos//I, pos % I

def multi_group_diffusion(R,I,groups,BC,diffusion,scattering,birth_rate,nu_Sig_fission,removal,geometry,epsilon=1.0e-8,LOUD=0,timer=0):
    import numpy as np
    ''' Solve a neutron diffusion eigenvalue problem in a 1-D geometry
    using cell-average unknowns
    Args: 
        R: size of domain
        I: number of cells
        groups: number of groups, (>2)
        BC: boundary conditions, must be in a (groups x 2) list
        diffusion: name of function that returns diffusion coefficient for given r
        scattering: name of function that returns macroscopic scattering for a given r
            must be in a (groups x groups) list and have zeros on the diagonal
        birth_rate: percentage of each group that is born fast, must be in list with sum = 1
        nu_Sig_fission: name of function that returns nu Sigma_fission for given r of group
        removal: name of function that returns the removal macroscopic cross-section for given r of group
        geometry: shape of problem
            0 for slab
            1 for cylindrical
            2 for spherical
    Returns:
        k: the multiplication factor of the system
        phi: array (I x group) of the flux from different groups
    '''
    
    Delta_r, centers, edges = create_grid(R,I) # Set up the Grid
    A = np.zeros((groups*(I+1),groups*(I+1))) # Left side matrix - removal cross-section
    B = np.zeros((groups*(I+1),groups*(I+1))) # Right side matrix - fission cross-section

    #define surface areas and volumes
    assert((geometry == 0) or (geometry == 1) or (geometry == 2)) #ensure geometry is correct
    if (geometry == 0): #slab
        # in slab it's 1 everywhere except at the left edge
        S = 0.0*edges+1 # surface area
        S[0] = 0.0 #to enforce Refl BC
        # in slab it's dr
        V = 0.0*centers +Delta_r #volume 
    elif (geometry == 1): #cylinder
        #i in cylinder it's 2pi r
        S = 2.0*np.pi*edges #surface area
        # in cylinder it's pi(r^2-r^2)
        V = np.pi*(edges[1:(I+1)]**2 - edges[0:I]**2) #volume
    elif (geometry == 2): #sphere
        # in sphere it's 4 pi^2
        S = 4.0*np.pi*edges**2 #surface area
        # in sphere its 4/3 pi(r^3-r^3)
        V = 4.0/3.0*np.pi*(edges[1:(I+1)]**3 - edges[0:I]**3) #volume
    
    # Left Hand Side
    # Diagonal Groups (gg)
    G = groups
    for g in range(groups):
        for i in range(I+1):
            if (i < I):
                r = centers[i] #determine the position for the inputs
                this_cell = get_pos_from_i_g(i,g,I+1,G) #move to a given group submatrix
                
                A[this_cell, this_cell] = (0.5/(Delta_r * V[i])*
                                           ((diffusion(r)[g]+diffusion(r+Delta_r)[g])*S[i+1])+
                                           removal(r)[g]) #move diagonally through the matrix
                A[this_cell, get_pos_from_i_g(i+1,g,I+1,G)] = -0.5*(diffusion(r)[g]+diffusion(r+Delta_r)[g])/(Delta_r*V[i])*S[i+1] #move column over one to the right(cell)
                if i > 0:
                    A[this_cell, get_pos_from_i_g(i-1,g,I+1,G)] = -0.5*(diffusion(r)[g]+diffusion(r-Delta_r)[g])/(Delta_r*V[i])*S[i] #move column over to the left (cell)
                    A[this_cell, this_cell] += 0.5/(Delta_r*V[i])*((diffusion(r)[g]+diffusion(r-Delta_r)[g])*S[i])
                #in scattering
                for gprime in range(groups):
                    if (gprime != g): #skip the same group scattering
                        A[this_cell, get_pos_from_i_g(i,gprime,I+1,G)] = -scattering(r)[g,gprime] #scattering diagonal
                    B[this_cell,get_pos_from_i_g(i,gprime,I+1,G)] = birth_rate(r)[g]*nu_Sig_fission(r)[gprime] #set up the fission diagonal 
            else: #sets the boundary conditions at the edge of each submatrix, i=I
                this_cell = get_pos_from_i_g(i,g,I+1,G) #move to a given submatrix
                A[this_cell,this_cell] = BC[g,0]*0.5+BC[g,1]/Delta_r #move to bottom right, boundary condition
                A[this_cell,get_pos_from_i_g(i-1,g,I+1,G)] = BC[g,0]*0.5-BC[g,1]/Delta_r #move one column to the left, boundary condition
                      
    # Solve for Eigenvalues, matrix equation set up
    phi,k = inversePower(I,A,B,groups,epsilon,LOUD=LOUD)
    
    # remove last row of phi
    phi = phi[:I]
    
    return k,phi

def group_matrix(R,I,groups,BC,diffusion,scattering,birth_rate,nu_Sig_fission,removal,geometry,epsilon=1.0e-8,LOUD=0,timer=0):
    import numpy as np
    ''' Solve a neutron diffusion eigenvalue problem in a 1-D geometry
    using cell-average unknowns
    Args: 
        Sig: SVD S matrix dimension
        R: size of domain
        I: number of cells
        groups: number of groups, (>2)
        BC: boundary conditions, must be in a (groups x 2) list
        diffusion: name of function that returns diffusion coefficient for given r
        scattering: name of function that returns macroscopic scattering for a given r
            must be in a (groups x groups) list and have zeros on the diagonal
        birth_rate: percentage of each group that is born fast, must be in list with sum = 1
        nu_Sig_fission: name of function that returns nu Sigma_fission for given r of group
        removal: name of function that returns the removal macroscopic cross-section for given r of group
        geometry: shape of problem
            0 for slab
            1 for cylindrical
            2 for spherical
    Returns:
        k: the multiplication factor of the system
        phi: array (I x group) of the flux from different groups
    '''
    
    Delta_r, centers, edges = create_grid(R,I) # Set up the Grid
    A = np.zeros((groups*(I+1),groups*(I+1))) # Left side matrix - removal cross-section
    B = np.zeros((groups*(I+1),groups*(I+1))) # Right side matrix - fission cross-section

    #define surface areas and volumes
    assert((geometry == 0) or (geometry == 1) or (geometry == 2)) #ensure geometry is correct
    if (geometry == 0): #slab
        # in slab it's 1 everywhere except at the left edge
        S = 0.0*edges+1 # surface area
        S[0] = 0.0 #to enforce Refl BC
        # in slab it's dr
        V = 0.0*centers +Delta_r #volume 
    elif (geometry == 1): #cylinder
        #i in cylinder it's 2pi r
        S = 2.0*np.pi*edges #surface area
        # in cylinder it's pi(r^2-r^2)
        V = np.pi*(edges[1:(I+1)]**2 - edges[0:I]**2) #volume
    elif (geometry == 2): #sphere
        # in sphere it's 4 pi^2
        S = 4.0*np.pi*edges**2 #surface area
        # in sphere its 4/3 pi(r^3-r^3)
        V = 4.0/3.0*np.pi*(edges[1:(I+1)]**3 - edges[0:I]**3) #volume
    
    # Left Hand Side
    # Diagonal Groups (gg)
    G = groups
    for g in range(groups):
        for i in range(I+1):
            if (i < I):
                r = centers[i] #determine the position for the inputs
                this_cell = get_pos_from_i_g(i,g,I+1,G) #move to a given group submatrix
                
                A[this_cell, this_cell] = (0.5/(Delta_r * V[i])*
                                           ((diffusion(r)[g]+diffusion(r+Delta_r)[g])*S[i+1])+
                                           removal(r)[g]) #move diagonally through the matrix
                A[this_cell, get_pos_from_i_g(i+1,g,I+1,G)] = -0.5*(diffusion(r)[g]+diffusion(r+Delta_r)[g])/(Delta_r*V[i])*S[i+1] #move column over one to the right(cell)
                if i > 0:
                    A[this_cell, get_pos_from_i_g(i-1,g,I+1,G)] = -0.5*(diffusion(r)[g]+diffusion(r-Delta_r)[g])/(Delta_r*V[i])*S[i] #move column over to the left (cell)
                    A[this_cell, this_cell] += 0.5/(Delta_r*V[i])*((diffusion(r)[g]+diffusion(r-Delta_r)[g])*S[i])
                #in scattering
                for gprime in range(groups):
                    if (gprime != g): #skip the same group scattering
                        A[this_cell, get_pos_from_i_g(i,gprime,I+1,G)] = -scattering(r)[g,gprime] #scattering diagonal
                    B[this_cell,get_pos_from_i_g(i,gprime,I+1,G)] = birth_rate(r)[g]*nu_Sig_fission(r)[gprime] #set up the fission diagonal 
            else: #sets the boundary conditions at the edge of each submatrix, i=I
                this_cell = get_pos_from_i_g(i,g,I+1,G) #move to a given submatrix
                A[this_cell,this_cell] = BC[g,0]*0.5+BC[g,1]/Delta_r #move to bottom right, boundary condition
                A[this_cell,get_pos_from_i_g(i-1,g,I+1,G)] = BC[g,0]*0.5-BC[g,1]/Delta_r #move one column to the left, boundary condition
    
    return A,B

# Taking SVD of phi matrices
def reduction(A,B,Sig):
    Au, As, Av = np.linalg.svd(A,full_matrices=True)
    Bu, Bs, Bv = np.linalg.svd(B,full_matrices=True)
    
    As[Sig:] = 0
    Bs[Sig:] = 0
    
    Reduc_SigA = np.diag(As)
    Reduc_SigB = np.diag(Bs)
    
    Reduc_A = Au.dot(Reduc_SigA.dot(Av))
    Reduc_B = Bu.dot(Reduc_SigB.dot(Bv))
    
    return Reduc_A,Reduc_B
