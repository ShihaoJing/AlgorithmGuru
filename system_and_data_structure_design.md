## 解题思路
* 分解问题的角度: fix 某一维度，尝试另一维度上的所有可能，可能是array（i，j）pointers
* 可能跟排序有关，尝试binary search
* 一旦需要统计一个元素在集合中出现的次数，尝试hashmap
* 遇到字符串，字典，charboard等，Trie tree总是可以试试的
* Range 里求最大/最小/Sum等特征，Segment Tree是个不错的选择
* Reversed idea非常重要, 最长可能是某种最短的反面, 最多可能是某种最少的反面, obstacle的反面是reachable, subarray的反面是array中的剩下元素, left的反面是right
* 在数组里找某一个数存不存在或者应该存在的位置时，注意number的取值范围，如果在0-n-1上时可能number本身就是index

## 经典题目

146. LRU Cache
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


460. LFU Cache
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


面经：Map with expired entry
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
            return this->expirationTime < n.expirationTime; 
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


379. Design Phone Directory
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

Google面经：Design HardDrive Management System, 给一个硬盘被分成N个区域，实现markUsed(int N) 和findNextNotUsed()

