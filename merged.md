## 解题思路
### General
* 分解问题的角度: fix 某一维度，尝试另一维度上的所有可能，可能是array（i，j）pointers

* 可能跟排序有关，尝试binary search

* 一旦需要统计一个元素在集合中出现的次数，尝试hashmap

* 求所有解，dfs + backtracking

* 遇到字符串，字典，charboard等，Trie tree总是可以试试的

* Range 里求最大/最小/Sum等特征，Segment Tree是个不错的选择

* Matrix和Array通常都是：1. Two Pointers，2. Sliding Window（fixed/not fixed），3. DP

* Reversed idea非常重要, 最长可能是某种最短的反面, 最多可能是某种最少的反面, obstacle的反面是reachable, subarray的反面是array中的剩下元素, left的反面是right

* 在数组里找某一个数存不存在或者应该存在的位置时，注意number的取值范围，如果在0-n-1上时可能number本身就是index

* 在做遍历时需要保存之前的信息，Stack是个不错的选择
LC 907, 496, 739

### Union Find 

Union Find主要解决的问题是dynamic connectivity，有N个object之间有connection时，Union Find可以帮助快速知道两个object是否有connection，
并且可以知道当前集合中有几个connected components.

Union Find的实现有以下几种：

1. Quick Find
在union(p, q)时，将所有root == root(p)的节点都指向root(q)
Time: union O(N), find O(1)
2. Quick Union
只将root(root(p))指向root(q)
Time: O(N) in worst case, Tree could get tall
3. Weighted Quick Union
Maintanin一个size数组, group with smaller size go under group with larger size
Time: O(logN), depth of each node is at most logN
4. Weighted Quick Union with Path Compression
Flatten Tree while finding the root of any node
TIme: close to linear  


### Tree
### Graph
### DP


#### Leetcode 11. Container With Most Water
<https://leetcode.com/problems/container-with-most-water/description/>

```c++
int maxArea(vector<int> &height) {
    int left = 0, right = height.size();
    int max_volume = 0;
    while (left < right) {
        max_volume = max(max_volume, min(height[left], height[right]) * (right - left));
        if (height[left] <= height[right]) {
            ++left;
        }
        else {
            --right;    
        }
    }
    return max_volume;
}
```

思路：Two pointers, 因为要得到更大的volume，必须要找higher height，而height由min(left, right)控制

#### Leetcode 42. Trapping Rain Water
<https://leetcode.com/problems/trapping-rain-water/description/>

```c++
int trap(vector<int>& height) {
    int water = 0;
    int left_most = 0, right_most = 0;
    int left = 0, right = height.size() - 1;
    while (left < right) {
        if (height[left] < height[right]) {
            if (height[left] > left_most) {
                left_most = height[left];
            }
            else {
                water += left_most - height[left];
            }
            ++left;
        }
        else {
            if (height[right] > right_most) {
                right_most = height[right];    
            }
            else {
                water += right_most - height[right];
            }
            --right;
        }
    }
    return water;
}
```

思路：Two pointers, 首先想到two pass dp，记录每一个位置当前左最高和右最高，之后想到one pass，只要知道当前left_most or
right_most

#### 259. 3Sum Smaller
<https://leetcode.com/problems/3sum-smaller/description/>

```c++
int threeSumSmaller(vector<int>& nums, int target) {
    sort(nums.begin(), nums.end());
    int res = 0;
    for (int i = 0; i < nums.size() - 2; ++i) {
        int j = nums.size() - 1;
        int k = i + 1;
        while (k < j) {
            int sum = nums[i] + nums[k] + nums[j];
            if (sum < target) {
                ++res;
                ++k;
            }
            else {
                --j;
            }
        }
    }
    return res;
}
```

思路：暴力解法是n三次方的复杂度，所以如果先排序，之后在找的时候可以知道方向，从而降到n方的复杂度


#### 41. First Missing Positive
<https://leetcode.com/problems/first-missing-positive/description/>

```c++
int firstMissingPositive(vector<int>& nums) {
    for (int i = 0; i < nums.size(); ++i) {
        while (nums[i] > 0 && nums[i] <= nums.size() && i != nums[i] - 1) {
            int temp = nums[nums[i-1] - 1];
            nums[nums[i] - 1] = nums[i];
            nums[i] = temp;
        }    
    }
    for (int i = 0; i < nums.size(); ++i) {
        if (i != nums[i] - 1) {
            return i + 1;    
        }    
    }
}

```
思路：输入范围是1-n，所以num i可以直接放在i - 1的位置，所以返回第一个i + 1 != nums[i]

#### 55. Jump Game I, II
<https://leetcode.com/problems/jump-game/description/>

```c++
bool canJump(vector<int>& nums) {
    int max_pos = nums[0];
    int i = 0;
    for (; i < nums.size() && i <= max_pos; ++i) {
        if (nums[i] + i > max_pos) {
            max_pos = max(max_pos, nums[i] + i);    
        }
    }
    return i == nums.size();
}
```

```c++
int canJump2(vector<int>& nums) {
    int step = 0;
    int max_pos = 0;
    int end = 0;
    for (int i = 0; i < nums.size(); ++i) {
        if (i > end) {
            ++step;
            end = max_pos;
        }
        max_pos = max(max_pos, nums[i] + i);
    }
    return step;
}
```

思路：知道了当前step能到达的最远距离，之后如果大于这个距离，则需要多一个step才能到达当前index


#### 35. Search Insert Position (with duplicates allowed)
<https://leetcode.com/problems/search-insert-position/description/>

```c++
int searchInsert(vector<int>& nums, int target) {
    int i = 0, j = nums.size() - 1;
    while (i <= j) {
        int mid = i + (j - i) / 2;    
        if (nums[mid] == target) {
            while (mid > 0 && nums[mid - 1] == target) { --mid; }    
            return mid;
        }
        else if (nums[mid] > target) {
            j = mid - 1;
        }
        els {
            i = mid + 1;
        }
    }

    return max(i, j);
}
```

思路：当target不存在时，if mid > target, j = j - 1, j < i, 此时返回的index就是mid的位置，反之则返回j

#### Google面经题

给定一个Quack class, 已经按升序排好， .pop() will randomly pop from head or tail of this data structure. 要求返利用这个.pop()，返回降序排列的 array。

```c++
// Quack::pop() will randomly pop from head or tail of a sorted data structure
vector<int> sort(Quack qk) {
    vector<int> sorted;
    stack<int> s;
    while (!qk.empty()) {
        int num = qk.pop();
        while (s.empty() || num > s.top()) {
            sorted.push_back(s.top());
            s.pop();
        }
        s.push(num);
    }
    while (!s.empty()) {
        sorted.push_back(s.top());
        s.pop();
    }

    return sorted;
}
```

思路：如果只从tail pop，用stack可以解，当从tail pop时，后面一定有大于top的，此时stack中小的数应该先加入sorted.

### 642. Design Search Autocomplete System
<https://leetcode.com/problems/design-search-autocomplete-system>

思路：由于要search with prefix，想到用trie，取最大的三个想到用priority queue,
improvment可以在search的过程中记录TrieNode的位置，下一次search时从当前TrieNode开始search

```c++
class AutocompleteSystem {
    class Trie {
        struct TrieNode {
            Node* children[26];
            int freq;
            bool isLeaf;
            TrieNode() : isLeaf(false), freq(0) {
                for (int i = 0; i < 26; ++i) {
                    children[i] = nullptr;
                }
            }
        };

        TrieNode *root;
        public:
        Trie(): root(new TrieNode()) { }
        void add(String word, int freq) {
            TrieNode *cur = root;
            for (char c: word) {
                if (cur->children[c-'a'] == nullptr) {
                    cur->children[c-'a'] = new TrieNode();
                }
                cur = cur->children[c-'a'];
            }
            cur->isLeaf = true;
            cur->freq = freq;
        }


        // 或者可以建立sentence->freq的map，update起来更快一些
        void update(String word) {
            TrieNode *cur = root;
            for (char c: word) {
                cur = cur->children[c-'a'];
            }
            ++cur->freq
        }

        vector<string> search(string prefix) {
            TrieNode *cur = root;
            string s;
            for (char c: word) {
                if (cur->children[c-'a'] == nullptr) { return vector<int>(); }
                s.push_back(c);
                cur = cur->children[c-'a'];
            }
            
            auto cmp = [](const pair<string, int> &a, const pair<string, int> &b) {
                return a.second > b.second || (a.second == b.second && a.first < b.first);
            };

            priority_queue<pair<string, int>, vector<pair<string, int>>, decltype(cmp)> q(cmp);

            dfs(cur, s, q);
            vector<string> res;
            while (!q.empty()) {
                res.push_back(q.top().first);
                q.pop();
            }
            return res;
        }

        void dfs(TrieNode *cur, string &s, priority_queue<...> &q) {
            if (cur == nullptr) { return; }
            if (cur->isLeaf) {
                q.push({s, cur->freq});
                if (q.size() > 3) {
                    q.pop();
                }
            }    
            for (int i = 0; i < 26; ++i) {
                s.push_back('a' + i);
                dfs(cur->children[i], s, q);
                s.pop_back();
            }
        }
    };

    Trie trie;
    string input;
public:
    AutocompleteSystem(vector<string> sentences, vector<int> times) {
        for (int i = 0; i < sentences; ++i) {
            trie.add(sentences[i], times[i]);
        } 
    }

    vector<string> input(char c) {
        if (c == '#') {
            // update freq of input string
            return {};    
        }
        data.push_back(c);
        return trie.search(data);
    }
};
```

#### Google
acba -》 dbcd，给一个start string 问能否转化成end string，在同一个时刻，只能转换一种字母，比如acba 可以转为dcbd （即把a变为d）
cb -> aa 返回true，c-> a, b-> a


#### 496. Next Greater Element I
<https://leetcode.com/problems/next-greater-element-i/description/>

思路：某一个位置之后的数如果对当前结果有影响，则可以从后到前遍历，用stack保持顺序

```c++
vector<int> nextGreaterElement(vector<int>& findNums, vector<int>& nums) {
        if (nums.size() == 0) return {};
        unordered_map<int, int> map;
        stack<int> s;
        s.push(nums.back());
        for (int i = nums.size() - 1; i >= 0; --i) {
            while (!s.empty() && nums[i] >= s.top()) {
                s.pop();
            }
            map[nums[i]] = s.empty() ? -1 : s.top();
            s.push(nums[i]);
        }
        vector<int> res;
        for (int num: findNums) {
            res.push_back(map[num]);
        }
        return res;
}
```

#### 739. Daily Temperatures

思路： 同上一题
```c++
vector<int> dailyTemperatures(vector<int>& temperatures) {
        vector<int> res(temperatures.size(), 0);
        stack<int> s;
        for (int i = temperatures.size() - 1; i >= 0; --i) {
            while (!s.empty() && temperatures[i] >= temperatures[s.top()]) {
                s.pop();
            }
            res[i] = s.empty() ? 0 : s.top() - i;
            s.push(i);
        }
        return res;
}
```

#### 79. Word Search
<https://leetcode.com/problems/word-search/description/>

思路：dfs + used map
```c++
bool exist(vector<vector<char>>& board, string word) {
    int m = board.size(), n = board[0].size();
    vector<vector<bool>> used(m, vector<bool>(n, false));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (board[i][j] == word[0] && dfs(board, used, i, j, 0, word)) {
                return true;
            }
        }
    }        
    return false;
}

bool dfs(vector<vector<char>> &board, vector<...> &used, int i, int j, int p, string word) {
    if (i < 0 || i >= board.size() || j < 0 || j >= board[i].size() || 
        used[i][j] ||
        board[i][j] != word[p]) {
        return false;
    }
    if (p == word.size() - 1) {
        return true;
    }

    used[i][j] = true;
    if (dfs(board, used, i+1, j, p+1, word) || dfs(board, used, i-1, j, p+1, word) ||
        dfs(board, used, i, j+1, p+1, word) || dfs(board, used, i, j-1, word)) {
        used[i][j] = false;
        return true;
    }

    used[i][j] = false;
    return false;
}
```

#### 212. Word Search II
<https://leetcode.com/problems/word-search-ii/description/>

思路：因为要提高查询效率，所以用Trie，只有当前prefix在Trie里才继续

```c++
class Solution {
    class Trie {
    public:
        struct Node {
            Node* children[26];
            string leaf;
        };

        Node *_root;
        /** Initialize your data structure here. */
        Trie() {
            _root = new Node();
        }

        /** Inserts a word into the trie. */
        void insert(string word) {
            Node *root = _root;
            for (int i = 0; i < word.length(); ++i) {
                int c = word[i] - 'a';
                if (root->children[c] == NULL) {
                    root->children[c] = new Node();
                }
                root = root->children[c];
            }
            root->leaf = word;
        }


    };
    Trie tree;
public:
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        for (string word: words) {
            tree.insert(word);
        }

        vector<string> res;

        for (int i = 0; i < board.size(); ++i) {
            for (int j = 0; j < board[i].size(); ++j) {
                dfs(board, res, i, j, tree._root);
            }
        }

        return res;
    }

    void dfs(vector<vector<char>> &board, vector<string> &res, int i, int j, Trie::Node *root) {
        if (i < 0 || i == board.size() || j < 0 || j == board[i].size() || board[i][j] == '.')
            return;


        char c = board[i][j];
        if (root->children[c - 'a'] == NULL) return;
        root = root->children[c - 'a'];
        if (root->leaf.length() > 0) {
            res.push_back(root->leaf); 
            root->leaf = string(); // avoid inserting duplicates
        }

        board[i][j] = '.';
        dfs(board, res, i+1, j, root);
        dfs(board, res, i-1, j, root);
        dfs(board, res, i, j+1, root);
        dfs(board, res, i, j-1, root);
        board[i][j] = c;
    }


};
```

#### 127. Word Ladder
<https://leetcode.com/problems/word-ladder/description/>

思路：两个单词之间的转换可以看成是graph里面的两个node，所以在graph里找最短路径想到用bfs

```c++
int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
    unordered_set<string> dict(wordList.begin(), wordList.end());
    unordered_set<string> used;
    int step = 0;
    queue<string> q;
    q.push_back(beginWord);
    while (!q.empty()) {
        ++step;
        int q_size = q.size();
        for (int i = 0; i < q_size(); ++i) {
            string candidate = q.front();
            q.pop();
            if (candidate == endWord) { return step; }
            if (used.find(candidate) != used.end()) { continue; }
            used.insert(candidate);
            for (int k = 0; k < candidate.size(); ++i) {
                string copy = candidate;
                for (int c = 0; c < 26; ++i) {
                   candidate[k] = 'a' + c;
                   if (dict.find(candidate) != dict.end()) {
                        q.push(candidate);
                    }
                }
                candidate = copy;
            }
        }
    }

    return 0;
}
```

#### 126. Word Ladder II
<https://leetcode.com/problems/word-ladder-ii/description/>

思路：bfs，保存path信息
```c++
class Solution {
public:
    std::vector<std::vector<std::string> > findLadders(std::string beginWord, std::string endWord, std::unordered_set<std::string> &dict) {
		std::vector<std::vector<std::string> > paths;
		std::vector<std::string> path(1, beginWord);
		if (beginWord == endWord) {
			paths.push_back(path);
			return paths;
		}
        std::unordered_set<std::string> words1, words2;
		words1.insert(beginWord);
		words2.insert(endWord);
		std::unordered_map<std::string, std::vector<std::string> > nexts;
		bool words1IsBegin = false;
        if (findLaddersHelper(words1, words2, dict, nexts, words1IsBegin))
			getPath(beginWord, endWord, nexts, path, paths);
		return paths;
    }
private:
    bool findLaddersHelper(
		std::unordered_set<std::string> &words1,
		std::unordered_set<std::string> &words2,
		std::unordered_set<std::string> &dict,
		std::unordered_map<std::string, std::vector<std::string> > &nexts,
		bool &words1IsBegin) {
		words1IsBegin = !words1IsBegin;
		if (words1.empty())
            return false;
		if (words1.size() > words2.size())
			return findLaddersHelper(words2, words1, dict, nexts, words1IsBegin);
		for (auto it = words1.begin(); it != words1.end(); ++it)
			dict.erase(*it);
		for (auto it = words2.begin(); it != words2.end(); ++it)
			dict.erase(*it);
        std::unordered_set<std::string> words3;
		bool reach = false;
        for (auto it = words1.begin(); it != words1.end(); ++it) {
			std::string word = *it;
			for (auto ch = word.begin(); ch != word.end(); ++ch) {
				char tmp = *ch;
                for (*ch = 'a'; *ch <= 'z'; ++(*ch))
					if (*ch != tmp)
						if (words2.find(word) != words2.end()) {
							reach = true;
							words1IsBegin ? nexts[*it].push_back(word) : nexts[word].push_back(*it);
						}
						else if (!reach && dict.find(word) != dict.end()) {
							words3.insert(word);
							words1IsBegin ? nexts[*it].push_back(word) : nexts[word].push_back(*it);
                        }
				*ch = tmp;
            }
        }
        return reach || findLaddersHelper(words2, words3, dict, nexts, words1IsBegin);
    }
	void getPath(
		std::string beginWord,
		std::string &endWord,
		std::unordered_map<std::string, std::vector<std::string> > &nexts,
		std::vector<std::string> &path,
		std::vector<std::vector<std::string> > &paths) {
		if (beginWord == endWord)
			paths.push_back(path);
		else
			for (auto it = nexts[beginWord].begin(); it != nexts[beginWord].end(); ++it) {
				path.push_back(*it);
				getPath(*it, endWord, nexts, path, paths);
				path.pop_back();
			}
	}
};
```

#### 215. Kth Largest Element in an Array
<https://leetcode.com/problems/kth-largest-element-in-an-array/description/>

解法1: Priority Queue
```c++
int findKthLargest(vector<int>& nums, int k) {
    priority_queue<int, vector<int>, greater<int>> pq;
    for (int num: nums) {
        pq.push(num);
        if (pq.size() > k) {
            pq.pop();
        }
    }
    return pq.top();
}
```

解法2: Quick Select
```c++
int quickSelect(vector<int> &nums, int k) {
        
}
```

#### 146. LRU Cache
<https://leetcode.com/problems/lru-cache/description/>

```c++
class LRUCache {
public:
    struct Node {
        int val;
        int key;
        Node *next;
        Node *pre;
        Node(int value, int k) : val(value), key(k), next(nullptr), pre(nullptr) { }
    };
    
    Node *head, *tail;
    unordered_map<int, Node*> cache;
    int cap;
    int size;
    
    void insert(Node *node) {
        node->next = head->next;
        head->next = node;
        node->next->pre = node;
        node->pre = head;
    }
    
    void remove(Node *node) {
        node->pre->next = node->next;
        node->next->pre = node->pre;
    }
    
    LRUCache(int capacity) {
        cap = capacity;
        size = 0;
        head = new Node(-1, -1);
        tail = new Node(-1, -1);
        head->next = tail;
        tail->pre = head;
    }
    
    int get(int key) {
        if (cache.find(key) == cache.end()) {
            return -1;
        }
        
        Node *node = cache[key];
        remove(node);
        insert(node);
        return node->val;
    }
    
    void put(int key, int value) {
        if (cache.find(key) == cache.end()) {
            Node *node = new Node(value, key);
            cache.insert({key, node});
            insert(node);
            if (++size > cap) {
                Node *tailNode = tail->pre;
                cache.erase(tailNode->key);
                remove(tailNode);
                --size;
            }
        }
        else {
            Node *node = cache[key];
            node->val = value;
            remove(node);
            insert(node);
        }
    }
};
```

思路：Map和Doubly Linked List的使用,
由于不需要知道某一个node具体用了多少次，只需要知道用的最少的那个，所以可以用插入排序的思想，然后用双链表实现


#### 460. LFU Cache
<https://leetcode.com/problems/lfu-cache/description/>

```c++
class LFUCache {
    int cap;
    int size;
    int minFreq;
    unordered_map<int, pair<int, int>> m; //key to {value,freq};
    unordered_map<int, list<int>::iterator> mIter; //key to list iterator;
    unordered_map<int, list<int>>  fm;  //freq to key list;
public:
    LFUCache(int capacity) {
        cap=capacity;
        size=0;
    }

    int get(int key) {
        if(m.count(key)==0) return -1;

        fm[m[key].second].erase(mIter[key]);
        m[key].second++;
        fm[m[key].second].push_back(key);
        mIter[key]=--fm[m[key].second].end();

        if(fm[minFreq].size()==0 )
              minFreq++;

        return m[key].first;
    }

   void put(int key, int value) {
        if(cap<=0) return;

        int storedValue=get(key);
        if(storedValue!=-1)
        {
            m[key].first=value;
            return;
        }

        if(size>=cap )
        {
            m.erase( fm[minFreq].front() );
            mIter.erase( fm[minFreq].front() );
            fm[minFreq].pop_front();
            size--;
        }

        m[key]={value, 1};
        fm[1].push_back(key);
        mIter[key]=--fm[1].end();
        minFreq=1;
        size++;
    }
};
```

思路：三个map，key->freq, key->node pointer, freq->key list, freqToKeyList的map里面实现了LRU


#### Google面经
Map with expired entry
实现一个Map Class，并实现这两个method
Integer get(String k)
    return value if key exists otherwise return null
void put(String k, int v, long ttl)
    ttl: time to live. The entry should expire beyond ttl

单线程
'''c++
class ExpirationMap {
    class Value {
        int val;
        long createdTime;
        long ttl;
    };  
    class Node {
        long expirationTime;
        int key;
        bool operator<(const Node &n) {
            return this->expirationTime > n.expirationTime; 
        }
    };
    unordered_map<int, Value> map;
    priority_queue<Node> expQueue;

    long getCurrentTime() { 
        // return current time in milliseconds (epoch time)
    }

    int get(int key) {
        if (map.find(key) == map.end()) { return -1; }
        Value v = map[key];
        if (v.createdTIme + v.ttl < getCurrentTime) {
            map.erase(key);
            return -1;
        }
        else {
            return v.val;
        }
    }

    void put(int key, int value, long ttl) {
        Value v(val, getCurrentTime(), ttl);
        map.insert({key, v});
        expQueue.push(Node(v.createdTIme + ttl, key));
    }

    // may be called in a deamon thread after a while
    void cleanUp() {
        while (1) {
            std::this_thread::sleep_for(1000); // sleep for 1s
            if (expQueue.size() > 0 && expQueue.top().expirationTime < getCurrentTime()) {
                map.erase(expQueue.top().key);
                expQueue.pop();
            }
        }
    }
};
```

多线程

```c++
#include <thread>
class Map {
    Map() {
        std::thread t(cleanup); 
        t.join();
    }   
};
```


### 379. Design Phone Directory
<https://leetcode.com/problems/design-phone-directory/description/>

思路：set与queue或者list, used集合可以用bitset减少memory使用
```c++
class PhoneDirectory {
public:
    /** Initialize your data structure here
        @param maxNumbers - The maximum numbers that can be stored in the phone directory. */
    queue<int> available;
    unordered_set<int> used;
    PhoneDirectory(int maxNumbers) {
        for (int i = 0; i < maxNumbers; ++i) {
            available.push(i);
        }
    }

    /** Provide a number which is not assigned to anyone.
        @return - Return an available number. Return -1 if none is available. */
    int get() {
        if (available.empty()) {
            return -1;
        }
        int next = available.front();
        available.pop();
        used.insert(next);
        return next;
    }

    /** Check if a number is available or not. */
    bool check(int number) {
        if (used.find(number) == used.end()) {
            return true;
        }

        return false;
    }

    /** Recycle or release a number. */
    void release(int number) {
        if (used.find(number) == used.end()) {
            return;
        }

        used.erase(number);
        available.push(number);
    }
};


// Your PhoneDirectory object will be instantiated and called as such:
// PhoneDirectory obj = new PhoneDirectory(maxNumbers);
// int param_1 = obj.get();
// bool param_2 = obj.check(number);
// obj.release(number);
```


### Google面经
Design HardDrive Management System, 给一个硬盘被分成N个区域，实现markUsed(int N) 和findNextNotUsed()

### Google面经
给一个 m * n 的board， 里面的值是0 或者1。 每一步我们可以去掉 一个点，如果这个点同一行行或者同一列列有其他的点。求一个remove order 使得我们可以去掉的点最多
1 0 0 
0 1 1  => 3 points 
1 0 0 
0 0 1 


分析：
    1. 从general case比较难得到解法，所以可以尝试一些特殊的输入，例如：
        1, 1, 1
        1, 1, 1  => 5 points

        1, 1, 0
        0, 1, 0  => 4 points
        0, 1, 1

       从特殊的case可以看出这是一个找connected component的问题，size为N的component最多可以remove N - 1个点
    2. 一般寻找connected component可以想到用dfs或者union find，如果要求输出某一个可行的sequence，则用dfs会简单很多

```c++
void dfs(vector<vector<int>> &chess, int i, int j, vector<pair<int, int>> &steps) {
    if (i < 0 || i >= chess.size() || j < 0 || j >= chess[i].size() || chess[i][j] == 0) {
        return;
    }

    chess[i][j] = 0;
    for (int k = j + 1; k < chess[i].size(); ++k) {
        if (chess[i][k] != 0) {
            dfs(chess, i, k, steps);
            break;
        }
    }
    for (int k = i + 1; k < chess.size(); ++k) {
        if (chess[k][j] != 0) {
            dfs(chess, k, j, steps);
            break;
        }
    }

    steps.push_back(pair<int, int>(i, j));
}

int find_max_removal(vector<vector<int>> &chess, int N) {
    int K = 0;
    vector<pair<int, int>> steps;
    for (int i = 0; i < chess.size(); ++i) {
        for (int j = 0; j < chess[i].size(); ++j) {
            if (chess[i][j] != 0) {
                ++K;
                dfs(chess, i, j, steps);
                steps.pop_back();
            }
        }
    }
    for (auto p: steps) {
        cout << p.first << " " << p.second << endl;
    }
    return N - K;
}
```

### 907. Sum of Subarray Minimums
<https://leetcode.com/problems/sum-of-subarray-minimums/description/>

分析：
    1. N方解法很容易想到
    2. 每一个数只在计算包含这个数并且这个数在当前subarray最小时有用，由此想到对于每一个数，如果知道这样的subarray的个数f(i)
       那么sum = f(i) * num(i)
    3. 如果知道以num(i)结尾的subarray的个数，和以num(i)开头的subarray的个数，那么f(i) = left(i) * right(i)
    4. 计算left(i)时，需要找到在num(i)左边，有几个数小于num(i)，由此可以想到用stack保存之前的信息, 并且stack需要保存对于num(i)的subarray的长度
    5. 注意连续的duplicates，只需要在left或者right中计算一次

    Time: O(n)
    Space: O(n)

```c++
int sumSubarrayMins(vector<int>& nums) {
    int sum = 0;
    int left[nums.size()], right[nums.size()];
    stack<pair<int, int>> s1, s2;
    for (int i = 0; i < nums.size(); ++i) {
        int count = 1;
        while (!s1.empty() && s1.top().first > nums[i]) {
            count += s1.top().second;
            s1.pop(); 
        }
        left[i] = count;
        s1.push({nums[i], count});
    }
    for (int i = 0; i < nums.size(); ++i) {
        int count = 1;
        while (!s2.empty() && s2.top().first >= nums[i]) { 
            count += s2.top().second;
            s2.pop(); 
        }
        right[i] = count;
        s2.push({nums[i], count});
    }
    for (int i = 0; i < nums.size(); ++i) {
        sum += nums[i] * left[i] * right[i];
        sum %= 1e9 + 7;     
    }
    return sum;
}
```

#### 399. Evaluate Division

分析：
    1. 由equation之间的关系可以看出这是一个graph，很容易想到用dfs找到路径
    2. dfs的复杂度可能是O(N2)的，由x/a = m, x/b = n, a/b = n/m这以性质可以想到用union find

DFS
```c++
 struct Edge {
     string n;
     double w;
     Edge(string node, double weight) : n(node), w(weight) {}
 };

 double dfs(unordered_map<string, vector<Edge>> &adjList, unordered_set<string> &visited, string node, string target) {
     if (node == target) {
         return 1.0;
     }

     if (visited.find(node) != visited.end()) {
         return -1.0;
     }

     visited.insert(node);

     for (auto &e: adjList[node]) {
         double res = dfs(adjList, visited, e.n, target);
         if (res != -1.0) {
             return res * e.w;
         }
     }

    return -1.0;
 }   
```

union find
```c++
class UnionFind {
public:
    unordered_map<string, string> root;
    unordered_map<string, double> ratio;
    unordered_map<string, int> rank;
    UnionFind() {}
    void addNode(string node) { 
        if (root.find(node) == root.end()) {
            root[node] = node;
            ratio[node] = 1.0;
            rank[node] = 1;
        }
    }

    void join(string p, string q, double r) {
        string root_p = find(p), root_q = find(q);
        if (root_p != root_q) {
            if (rank[root_p] >= rank[root_q]) {
                root[root_q] = root_p;
                ratio[root_q] = r * ratio[p];
                ++rank[root_p];
            }
            else {
                root[root_p] = root_q;
                ratio[root_p] = 1.0 / r * ratio[q];
                ++rank[root_q];
            }
        }
    }

    string find(string p) {
        if (root[p] != p) {
            string r = find(root[p]);
            ratio[p] *= ratio[root[p]];
            root[p] = r;
        }
        return root[p];
    }
};

vector<double> calcEquation(vector<pair<string, string>> equations, vector<double>& values, vector<pair<string, string>> queries) {
    UnionFind uf;
    for (int i = 0; i < equations.size(); ++i) {
        auto &e = equations[i];
        uf.addNode(e.first);
        uf.addNode(e.second);
        uf.join(e.first, e.second, values[i]);
    }

    vector<double> res;
    for (auto &q: queries) {
        if (uf.root.find(q.first) == uf.root.end() || uf.root.find(q.second) == uf.root.end()) {
            res.push_back(-1);
        }
        else if (q.first == q.second) {
            res.push_back(1);
        }
        else {
            string root_p = uf.find(q.first), root_q = uf.find(q.second);
            if (root_p != root_q) {
                res.push_back(-1); 
            }
            else {
                //cout << uf.ratio[q.second] << endl;
                //cout << uf.ratio[q.first] << endl;
                res.push_back(uf.ratio[q.second] / uf.ratio[q.first]);
            }
        }
    }

    return res;
}
```
