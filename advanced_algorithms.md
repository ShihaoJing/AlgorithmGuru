## Segment Tree

Basic Implementation

```c++
class SegmentTree {
    vector<int> data;
    int n; 
    void buildTree(vector<int> &nums, int cur, int l, int r) {
        if (l == r) {
            data[cur] = nums[l];
            return data[cur];
        }
        int mid = l + (r - l) / 2;
        int left = buildTree(nums, 2*cur, l, mid);
        int right = buildTree(nums, 2*cur+1, mid+1, right);
        data[cur] = left + right;
        return data[cur];
    }
public:
    SegmentTree(vector<int> nums) {
        n = nums.size();
        // nums的size为叶节点的数量
        // 其实是full binary tree 
        // 但是此处用complete binary tree大小的数组写起来更方便
        int max_size = pow(2, 1 + int(ceil(log2(n)))); 
        data = vector<int>(0, max_size);
        buildTree(nums, 1, 0, n - 1);
    }

    int sumRange(int i, int j) {
        return sumRangeHelper(1, 0, N - 1, i, j); 
    }

    int sumRangeHelper(int cur, int l, int r, int i, int j) {
        if (l > j || r < i) { return 0; }
        else if (l >= i && r <= j) { return data[cur]; }
        else {
            mid = l + (r - l) / 2;
            return sumRangeHelper(cur*2, l, mid, i, j) + sumRangeHelper(cur*2+1, mid+1, r, i, j);
        }
    
    }

    void update(int i, int val) {
        updateHelper(1, 0, N-1, i, val); 
    }

    void updateHelper(int cur, int l, int r, int pos, int val) {
        if (l == r) {
            data[cur] = val;
            return data[cur];
        }
        mid = l + (r - l) / 2;
        if pos <= mid:
            updateHelper(cur*2, l, mid, val);
        else:
            updateHelper(cur*2+1, mid+1, r, val);

        data[cur] = data[cur*2] + data[cur*2+1];
    }

};
```


## Union Find

Path Compression is implemented by default.

Basic Implementation

```c++
class disjointset {
    vector<int> root;
    disjointset(int n) {
        root = vector<int>(n);
        for (int i = 0; i < n; ++i) {
            root[i] = i;
        }
    }
    void union(int p, int q) {
        int root_p = find(p), root_q = find(q);
        if (root_p != root_q) {
            root[root_p] = root_q;
        }
    }

    int find(int p) {
        if (root[p] != p) {
            root[p] = find(root[p]);
        }
        return root[p];
    }
};
```

Uninon Find with Rank/Size

```c++
class disjointset {
    vector<int> root;
    vector<int> rank;
    disjointset(int n) {
        root = vector<int>(n);
        rank = vector<int>(n, 1);
        for (int i = 0; i < n; ++i) {
            root[i] = i;
        }
    }
    void union(int p, int q) {
        int root_p = find(p), root_q = find(q);
        if (root_p != root_q) {
            if (rank[root_p] < rank[root_q]) {
                root[root_p] = root_q;
            }
            else if (rank[root_p] > rank[root_q]) {
                root[root_q] = root_p;
            }
            else {
                root[root_q] = root_p;
                ++rank[root_p];
            }
        }
    }

    int find(int p) {
        if (root[p] != p) {
            root[p] = find(root[p]);
        }
        return root[p];
    }
};
```

## KMP Algorithm

## Dijkstra Algorithm

## MST Algorithms

### Prim MST
### Kruskal MST


## A-star Algorithm
