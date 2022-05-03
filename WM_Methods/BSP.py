import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools

# Define the BSP function that is called and recursively subdivides a 2D distribution
def calc(x, y, z, depth: int, axis: int = 0, **kwargs):
    """
    Create a binary space partition tree from a set of point coordinates

    x = np.array([0,1,2,3,4])
    y = np.array([0,1,2,3,4])
    z = np.array([0,1,2,3,4])
    a = np.array([0,1,2,3,4])
    b = np.array([0,1,2,3,4])
    c = np.array([0,1,2,3,4])
    d = np.array([0,1,2,3,4])
    e = np.array([0,1,2,3,4])

    binary_space_partition(x,y,z, depth=0,axis=0, sum=[a,b], mean=[c,d], weight=e)

    Args:
        x: x coordinates
        y: y coordinates
        z: distribution data (e.g. volume)
        **kwargs:
        'sum': variables of interest to integrate in a BSP bin (e.g. total volume/surface area) [max 6]
        'mean': variables of interest to distribution-weighted average in a BSP bin (e.g. volume-averaged carbon) [max 6]
        'weight': variable over which to weight the 'mean' variables [max 1]
        depth: maximum tree depth
        axis: initial branch axis
        
    Returns:
        A tuple (bounds, (left, right)) where bounds is the bounding
        box of the contained points and left and right are the results
        of calling binary_space_partition on the left and right tree
        branches. Once the tree has reached 'depth' levels then the
        second element will be None
    """
    
    ### Take all sum, mean and weight arrays
    names = list(kwargs.keys())

    if names[0] == 'sum':
        sum_vals = np.array(list(kwargs.values())[0])
    elif names[1] == 'sum':
        sum_vals = np.array(list(kwargs.values())[1])
    elif names[2] == 'sum':
        sum_vals = np.array(list(kwargs.values())[2])

    if names[0] == 'mean':
        mean_vals = np.array(list(kwargs.values())[0])
    elif names[1] == 'mean':
        mean_vals = np.array(list(kwargs.values())[1])
    elif names[2] == 'mean':
        mean_vals = np.array(list(kwargs.values())[2])
    
    if names[0] == 'weight':
        weight_vals = np.array(list(kwargs.values())[0])
    elif names[1] == 'weight':
        weight_vals = np.array(list(kwargs.values())[1])
    elif names[2] == 'weight':
        weight_vals = np.array(list(kwargs.values())[2])
    
    w = weight_vals
    wsum = w.sum()

    if (names[0] != 'sum') & (names[1] != 'sum') & (names[2] != 'sum'):
        print('ERROR: Summing variables must be provided. Pass at least one for analysis.')
        return
    if (names[0] != 'mean') & (names[1] != 'mean') & (names[2] != 'mean'):
        print('ERROR: Mean variables must be provided. Pass at least one for analysis.')
        return
    if (names[0] != 'weight') & (names[1] != 'weight') & (names[2] != 'weight'):
        print('ERROR: Weights must be provided. Pass array of ones for unweighted analysis.')
        return

    if len(sum_vals.shape) == 1:
        print('ERROR: sum variables must be passed as a list of arrays')
        return
    if len(mean_vals.shape) == 1:
        print('ERROR: mean variables must be passed as a list of arrays')
        return
    if sum_vals.shape[0] >6:
        print('ERROR: Function does not support more than 6 variables to sum. Reduce number of variables to <=6')
        return
    if mean_vals.shape[0] >6:
        print('ERROR: Function does not support more than 6 variables to mean. Reduce number of variables to <=6')
        return

    if len(w.shape) >=2:
        print('ERROR: Function only supports one weight variable. Reduce number of variables to 1')
        print('Current weight variables =', len(w.shape))
        return
    
    ### Calculate diagnostics for sum, mean and weight arrays

    if sum_vals.shape[0] ==6:
        a = sum_vals[0]
        b = sum_vals[1]
        c = sum_vals[2]
        d = sum_vals[3]
        e = sum_vals[4]
        f = sum_vals[5]
        asum = a.sum()
        bsum = b.sum()
        csum = c.sum()
        dsum = d.sum()
        esum = e.sum()
        fsum = f.sum()
        sum_list = np.array([asum,bsum,csum,dsum,esum,fsum])
    elif sum_vals.shape[0] ==5:
        a = sum_vals[0]
        b = sum_vals[1]
        c = sum_vals[2]
        d = sum_vals[3]
        e = sum_vals[4]
        asum = a.sum()
        bsum = b.sum()
        csum = c.sum()
        dsum = d.sum()
        esum = e.sum()
        sum_list = np.array([asum,bsum,csum,dsum,esum])
    elif sum_vals.shape[0] ==4:
        a = sum_vals[0]
        b = sum_vals[1]
        c = sum_vals[2]
        d = sum_vals[3]
        asum = a.sum()
        bsum = b.sum()
        csum = c.sum()
        dsum = d.sum()
        sum_list = np.array([asum,bsum,csum,dsum])

    elif sum_vals.shape[0] ==3:
        a = sum_vals[0]
        b = sum_vals[1]
        c = sum_vals[2]
        asum = a.sum()
        bsum = b.sum()
        csum = c.sum()
        sum_list = np.array([asum,bsum,csum])

    elif sum_vals.shape[0] ==2:
        a = sum_vals[0]
        b = sum_vals[1]
        asum = a.sum()
        bsum = b.sum()
        sum_list = np.array([asum,bsum])
    elif sum_vals.shape[0] ==1:
        a = sum_vals[0]
        asum = a.sum()
        sum_list = np.array([asum])

    if mean_vals.shape[0] ==6:
        g = mean_vals[0]
        h = mean_vals[1]
        i = mean_vals[2]
        j = mean_vals[3]
        k = mean_vals[4]
        l = mean_vals[5]
        gmean = (g*w).sum()/wsum
        hmean = (h*w).sum()/wsum
        imean = (i*w).sum()/wsum
        jmean = (j*w).sum()/wsum
        kmean = (k*w).sum()/wsum
        lmean = (l*w).sum()/wsum
        mean_list = np.array([gmean,hmean,imean,jmean,kmean,lmean])

    elif mean_vals.shape[0] ==5:
        g = mean_vals[0]
        h = mean_vals[1]
        i = mean_vals[2]
        j = mean_vals[3]
        k = mean_vals[4]
        gmean = (g*w).sum()/wsum
        hmean = (h*w).sum()/wsum
        imean = (i*w).sum()/wsum
        jmean = (j*w).sum()/wsum
        kmean = (k*w).sum()/wsum
        mean_list = np.array([gmean,hmean,imean,jmean,kmean])
    elif mean_vals.shape[0] ==4:
        g = mean_vals[0]
        h = mean_vals[1]
        i = mean_vals[2]
        j = mean_vals[3]
        gmean = (g*w).sum()/wsum
        hmean = (h*w).sum()/wsum
        imean = (i*w).sum()/wsum
        jmean = (j*w).sum()/wsum
        mean_list = np.array([gmean,hmean,imean,jmean])
    elif mean_vals.shape[0] ==3:
        g = mean_vals[0]
        h = mean_vals[1]
        i = mean_vals[2]
        gmean = (g*w).sum()/wsum
        hmean = (h*w).sum()/wsum
        imean = (i*w).sum()/wsum
        mean_list = np.array([gmean,hmean,imean])
    elif mean_vals.shape[0] ==2:
        g = mean_vals[0]
        h = mean_vals[1]
        gmean = (g*w).sum()/wsum
        hmean = (h*w).sum()/wsum
        mean_list = np.array([gmean,hmean])
    elif mean_vals.shape[0] ==1:
        g = mean_vals[0]
        gmean = (g*w).sum()/wsum
        mean_list = np.array([gmean])
    
    bounds = (x.min(), x.max(), y.min(), y.max())
    
    if depth == 0 or x.size <= 2:
        # Add diagnostic to  the output
        return [bounds, sum_list, mean_list, None]
    
    # Sort coordinates along axis
    if axis == 0:
        idx = np.argsort(x)
    elif axis == 1:
        idx = np.argsort(y)
    else:
        raise ArgumentError
    
    # Indexes for left and right branches
    # Use volume on current branch to find the split at the centre point in volume
    vtot_half = z.sum()/2.
    v1 = z[idx].cumsum()
    
    idx_l = idx[v1<vtot_half]
    idx_r = idx[v1>vtot_half]
    
    # Recurse into the branches
    left = calc(x[idx_l], y[idx_l], z[idx_l], depth-1, (axis+1)%2, sum = sum_vals[:,idx_l], mean = mean_vals[:,idx_l], weight = w[idx_l])  
    right = calc(x[idx_r], y[idx_r], z[idx_r], depth-1, (axis+1)%2, sum = sum_vals[:,idx_r], mean = mean_vals[:,idx_r], weight = w[idx_r])

    result = [left, right]

    return result

def split(bsp, depth : int):

    result_flat = list(itertools.chain(*bsp))
    while (len(result_flat) <= 2**depth):
        result_flat = list(itertools.chain(*result_flat))

    box_bounds = np.array(result_flat[::4])
    summed_vars = np.array(result_flat[1::4])
    meaned_vars = np.array(result_flat[2::4])

    return {'bounding_box':box_bounds, 'summed_vals':summed_vars, 'meaned_vals':meaned_vars}

def draw(x,y,z, partitions, edge_color, depth:int, **kwargs):
    """
    Plot the bounding boxes of binary space partition 'partitions' at 'depth'
    """
    # Plot this level
    plt.scatter(x,y,1,z, **kwargs)
    for i in range(2**depth):
        plt.gca().add_patch(patches.Rectangle((partitions[i,0], partitions[i,2]), partitions[i,1]-partitions[i,0], partitions[i,3]-partitions[i,2], ec=edge_color, facecolor='none'))
    