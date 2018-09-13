## 解题思路
* 分解问题的角度: fix 某一维度，尝试另一维度上的所有可能，可能是array（i，j）pointers
* 可能跟排序有关，尝试binary search
* 一旦需要统计一个元素在集合中出现的次数，尝试hashmap
* 求所有解，dfs + backtracking
* 遇到字符串，字典，charboard等，Trie tree总是可以试试的
* Range 里求最大/最小/Sum等特征，Segment Tree是个不错的选择
* Matrix和Array通常都是：1. Two Pointers，2. Sliding Window（fixed/not fixed），3. DP
* Reversed idea非常重要, 最长可能是某种最短的反面, 最多可能是某种最少的反面, obstacle的反面是reachable, subarray的反面是array中的剩下元素, left的反面是right
* 在数组里找某一个数存不存在或者应该存在的位置时，注意number的取值范围，如果在0-n-1上时可能number本身就是index

## 经典题型

### Two pointers
Leetcode 11. Container With Most Water
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

Leetcode 42. Trapping Rain Water
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

259. 3Sum Smaller
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


41. First Missing Positive
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

55. Jump Game I, II
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


### Binary Search

* Binary Search一般与排序或部分排序的搜索有关
* i, j在while循环的判断条件有两种，i < j or i <= j，大部分情况下用i <= j比较好写，此时解在while循环里返回，return
  invalid条件
* mid用i + (j - i) / 2避免int overflow

35. Search Insert Position (with duplicates allowed)
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

Google面经题：给定一个Quack class, 已经按升序排好， .pop() will randomly pop from head or tail of this data structure. 要求返利用这个.pop()，返回降序排列的 array。

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

### Trie 

基本结构 (only 26 lower case letter)
```c++
class Trie {
    struct TrieNode {
        Node* children[26];
        bool isLeaf;
        TrieNode() : isLeaf(false) {
            for (int i = 0; i < 26; ++i) {
                children[i] = nullptr;
            }
        }
    };

    TrieNode *root;
public:
    Trie(): root(new TrieNode()) { }
    void add(String word) {
        TrieNode *cur = root;
        for (char c: word) {
            if (cur->children[c-'a'] == nullptr) {
                cur->children[c-'a'] = new TrieNode();
            }
            cur = cur->children[c-'a'];
        }
        cur->isLeaf = true;
    }

    bool search(string word) {
        TrieNode *cur = root;
        for (char c: word) {
            if (cur->children[c-'a'] == nullptr) { return false; }
            cur = cur->children[c-'a'];
        }
        return cur->isLeaf;
    }
};
```

642. Design Search Autocomplete System
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

Google面经：acba -》 dbcd，给一个start string 问能否转化成end string，在同一个时刻，只能转换一种字母，比如acba 可以转为dcbd （即把a变为d）
cb -> aa 返回true，c-> a, b-> a

## 与Next有关的问题

496. Next Greater Element I
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

739. Daily Temperatures

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
