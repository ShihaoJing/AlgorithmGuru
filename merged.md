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


/**
 * Your PhoneDirectory object will be instantiated and called as such:
 * PhoneDirectory obj = new PhoneDirectory(maxNumbers);
 * int param_1 = obj.get();
 * bool param_2 = obj.check(number);
 * obj.release(number);
 */

```


### Google面经
Design HardDrive Management System, 给一个硬盘被分成N个区域，实现markUsed(int N) 和findNextNotUsed()