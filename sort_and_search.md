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

79. Word Search
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

212. Word Search II
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

127. Word Ladder
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

126. Word Ladder II
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
