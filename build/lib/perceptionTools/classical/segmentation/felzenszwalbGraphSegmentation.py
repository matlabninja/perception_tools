import numpy as np
import scipy
import random

class felzenGraphSegmenter:
    # Initialize the class
    def __init__(self, sigma: float=0.8, k: int = 300, method: str = 'grid',
                 single_channel: bool = True) -> None:
        # Set the parameters up
        self.k = k
        self.sigma = sigma
        self.method = method
        self.single_channel = single_channel

    def segment_image(self, img: np.ndarray) -> dict:
        # Convert to FP32 so you can do more calculations on this
        img = img.astype(np.float32)
        # Collapse to single image channel if called for
        if self.single_channel:
            img = self.collapse_image(img)
        # Apply gaussian filter
        if self.sigma > 0:
            img = scipy.ndimage.gaussian_filter(img, self.sigma)
        # Compute image graph
        if self.method == 'grid':
            graph_rep = self.compute_image_graph_grid(img)
        else:
            raise NotImplementedError("Nearest neighbor feature method not implemented")
        # Segment the graph
        return self.segment_graph(graph_rep,img.shape)

    def compute_image_graph_grid(self,img: np.ndarray) -> np.ndarray:
        # Compute the dissimilarity maps
        # Set up filters
        hshiftneg = [0,0,0,1]
        vshift = [0,1,1,1]
        hshift = [1,0,1,0]
        kernels = [np.array([[1,-1]]),np.array([[1],[-1]]),np.array([[1,0],[0,-1]]),np.array([[0,1],[-1,0]])]
        maps = []
        for k in kernels:
            curr_map = None
            if len(img.shape) == 2:
                chan_count = 1
            else:
                chan_count = img.shape[2]
            for chan in range(chan_count):
                if len(img.shape) == 2:
                    map_part = scipy.signal.convolve2d(img, k, mode='valid')
                else:
                    map_part = scipy.signal.convolve2d(img[:,:,chan], k, mode='valid')
                if curr_map is None:
                    curr_map = np.power(map_part,2)
                else:
                    curr_map += np.power(map_part,2)
            maps += [np.sqrt(curr_map)]

        # Set up indexing arrays and difference data for algo, compute and stack graph map
        graph_stack = []
        for idx,curr_map in enumerate(maps):
            # Create indexing maps
            vInd = np.arange(curr_map.shape[0])
            hInd = np.arange(curr_map.shape[1])
            hMap,vMap = np.meshgrid(hInd,vInd)
            # Flatten indexing maps for graph array
            hMap = hMap.reshape((1,-1))
            vMap = vMap.reshape((1,-1))
            # Flatten dissimilarity map for graph array
            curr_map = curr_map.reshape((1,-1))
            # Stack
            graph_stack += [np.vstack([curr_map,vMap,hMap+hshiftneg[idx],vMap+vshift[idx],hMap+hshift[idx]])]
        graph_stack = np.hstack(graph_stack)
        sort_index = np.argsort(graph_stack[0,:])
        graph_stack = graph_stack[:,sort_index]
        return graph_stack
    
    def segment_graph(self,graph_stack: np.array, img_shape: tuple[int,int]) -> dict:
        # Build initial component sets
        region_sets = {}
        region_conn = {}
        ind = 0
        # This array gets used as a reverse lookup for tracking pixel region membership
        rev_lookup = -1*np.ones((img_shape[0],img_shape[1])).astype(np.uint32)
        # Create component sets, max in-component dissimilarity tracker, and reverse lookup image
        for idx in range(img_shape[0]):
            for idx2 in range(img_shape[1]):
                region_sets[ind] = {(idx,idx2)}
                region_conn[ind] = 0
                rev_lookup[idx,idx2] = ind
                ind += 1
        # Start merging regions
        for idx in range(graph_stack.shape[1]):
            # Get regions that these are part of
            pr0 = rev_lookup[int(graph_stack[1,idx]),int(graph_stack[2,idx])]
            pr1 = rev_lookup[int(graph_stack[3,idx]),int(graph_stack[4,idx])]
            if pr0 == pr1:
                continue
            # Pull dissimilarity
            mc0 = region_conn[pr0]
            mc1 = region_conn[pr1]
            # Compute threshold functions
            t0 = mc0 + self.k/len(region_sets[pr0])
            t1 = mc1 + self.k/len(region_sets[pr1])
            # Check for merge
            if graph_stack[0,idx] <= t0 and graph_stack[0,idx] <= t1:
                region_sets[pr0] = region_sets[pr0].union(region_sets[pr1])
                region_conn[pr0] = max(graph_stack[0,idx],max(region_conn[pr0],region_conn[pr1]))
                rev_lookup[rev_lookup == pr1] = pr0
                del(region_sets[pr1])
                del(region_conn[pr1])
        # Return the segmentation map
        return region_sets
    
    # Collapse image to single channel
    def collapse_image(self,img):
        return np.mean(img,axis=2)
    
    # Convert region set output to image
    @staticmethod
    def region_set_to_img(region_sets: dict, img_shape: tuple[int,int]) -> np.ndarray:
        seg_img = np.zeros((img_shape[0],img_shape[1],3)).astype(np.uint8)
        for key,val in region_sets.items():
            r = random.randrange(256)
            g = random.randrange(256)
            b = random.randrange(256)
            for coord in val:
                seg_img[coord[0],coord[1],0] = r
                seg_img[coord[0],coord[1],1] = g
                seg_img[coord[0],coord[1],2] = b
        return seg_img