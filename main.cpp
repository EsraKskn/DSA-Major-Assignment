//
//  main.cpp
//  majorAssignment
//
//  Created by Esra Keskin on 11.05.2024.
//

/*
//1.problem
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// Structure to represent a meeting
struct Meeting {
    int start;
    int end;
    int index; // To keep track of original index

    Meeting(int s, int e, int i) : start(s), end(e), index(i) {}
};

// Comparator to sort meetings based on end times
bool compare(const Meeting& a, const Meeting& b) {
    return a.end < b.end;
}

// Function to find maximum number of meetings
int maxMeetings(int start[], int end[], int N) {
    // Create a vector of meetings
    vector<Meeting> meetings;
    for (int i = 0; i < N; i++) {
        meetings.push_back(Meeting(start[i], end[i], i));
    }

    // Sort meetings based on end times
    sort(meetings.begin(), meetings.end(), compare);

    // Select the first meeting
    int count = 1;
    int endTime = meetings[0].end;

    // Select subsequent meetings if their start time is after the end time
    for (int i = 1; i < N; i++) {
        if (meetings[i].start > endTime) {
            count++;
            endTime = meetings[i].end;
        }
    }

    return count;
}

int main() {
    int N = 6;
    int start[] = {1, 3, 0, 5, 8, 5};
    int end[] = {2, 4, 6, 7, 9, 9};

    cout << "Maximum number of meetings: " << maxMeetings(start, end, N) << endl;

    return 0;
}
*/

/*
//2.problem
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// Structure to represent a job
struct Job {
    int id;
    int deadline;
    int profit;
};

// Function to compare jobs based on profit in decreasing order
bool compare(Job& a, Job& b) {
    return a.profit > b.profit;
}

// Function to find maximum profit and number of jobs done
pair<int, int> maxProfit(vector<Job>& jobs) {
    // Sort jobs based on profit in decreasing order
    sort(jobs.begin(), jobs.end(), compare);

    int maxProfit = 0;
    int numJobsDone = 0;

    // Find maximum profit and number of jobs done
    vector<bool> slot(jobs.size(), false);
    for (int i = 0; i < jobs.size(); ++i) {
        for (int j = min(jobs[i].deadline - 1, (int)jobs.size() - 1); j >= 0; --j) {
            if (!slot[j]) {
                slot[j] = true;
                maxProfit += jobs[i].profit;
                ++numJobsDone;
                break;
            }
        }
    }

    return {numJobsDone, maxProfit};
}

int main() {
    // Example 1
    int N1 = 4;
    vector<Job> jobs1 = {{1, 4, 20}, {2, 1, 10}, {3, 1, 40}, {4, 1, 30}};

    pair<int, int> result1 = maxProfit(jobs1);

    cout << "Example 1:" << endl;
    cout << "Number of jobs done: " << result1.first << endl;
    cout << "Maximum profit: " << result1.second << endl;

    // Example 2
    int N2 = 5;
    vector<Job> jobs2 = {{1, 2, 100}, {2, 1, 19}, {3, 2, 27}, {4, 1, 25}, {5, 1, 15}};

    pair<int, int> result2 = maxProfit(jobs2);

    cout << "\nExample 2:" << endl;
    cout << "Number of jobs done: " << result2.first << endl;
    cout << "Maximum profit: " << result2.second << endl;

    return 0;
}
*/

/*
//3.problem
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

struct Job {
    int id;
    int deadline;
    int profit;
};

bool compare(Job a, Job b) {
    return (a.profit > b.profit);
}

pair<int, int> findMaxProfit(vector<Job>& jobs) {
    sort(jobs.begin(), jobs.end(), compare);

    int maxDeadline = 0;
    for (const Job& job : jobs) {
        maxDeadline = max(maxDeadline, job.deadline);
    }

    vector<int> result(maxDeadline, -1);
    int count = 0, profit = 0;

    for (int i = 0; i < jobs.size(); ++i) {
        for (int j = jobs[i].deadline - 1; j >= 0; --j) {
            if (result[j] == -1) {
                result[j] = jobs[i].id;
                count++;
                profit += jobs[i].profit;
                break;
            }
        }
    }

    return make_pair(count, profit);
}

int main() {
    vector<Job> jobs = {{1, 4, 20}, {2, 1, 10}, {3, 1, 40}, {4, 1, 30}};
    pair<int, int> result = findMaxProfit(jobs);
    cout << "Number of jobs done: " << result.first << endl;
    cout << "Maximum profit: " << result.second << endl;

    return 0;
}
*/

/*
//4.problem
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Pair {
    int first;
    int second;
};

bool comparePairs(const Pair &a, const Pair &b) {
    return a.second < b.second;
}

int maxChainLength(vector<Pair>& pairs) {
    if (pairs.empty()) return 0;
    
    sort(pairs.begin(), pairs.end(), comparePairs);

    int maxChainLen = 1;
    int currentEnd = pairs[0].second;

    for (int i = 1; i < pairs.size(); ++i) {
        if (pairs[i].first > currentEnd) {
            maxChainLen++;
            currentEnd = pairs[i].second;
        }
    }

    return maxChainLen;
}

int main() {
    vector<Pair> pairs = {{5, 24}, {39, 60}, {15, 28}, {27, 40}, {50, 90}};
    cout << "Max chain length for first example: " << maxChainLength(pairs) << endl;

    pairs = {{5, 10}, {1, 11}};
    cout << "Max chain length for second example: " << maxChainLength(pairs) << endl;

    return 0;
}
*/

/*
//5.problem
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Pair {
    int first;
    int second;
};

bool comparePairs(const Pair &a, const Pair &b) {
    return a.second < b.second;
}

int maxChainLength(vector<Pair>& pairs) {
    if (pairs.empty()) return 0;

    sort(pairs.begin(), pairs.end(), comparePairs);

    int n = pairs.size();
    vector<int> dp(n, 1);

    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (pairs[i].first > pairs[j].second) {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
    }

    return *max_element(dp.begin(), dp.end());
}

int main() {
    vector<Pair> pairs1 = {{5, 24}, {39, 60}, {15, 28}, {27, 40}, {50, 90}};
    cout << "Max chain length for first example: " << maxChainLength(pairs1) << endl;

    vector<Pair> pairs2 = {{5, 10}, {1, 11}};
    cout << "Max chain length for second example: " << maxChainLength(pairs2) << endl;

    return 0;
}
*/

/*
//6.problem
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

void findPaths(vector<vector<int>>& maze, int x, int y, int N, string path, vector<string>& result) {
    // Base case: if current position is the destination
    if (x == N - 1 && y == N - 1) {
        result.push_back(path);
        return;
    }

    // Mark current cell as visited
    maze[x][y] = 0;

    // Possible moves: up, down, left, right
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};
    char directions[] = {'U', 'D', 'L', 'R'};

    // Try all possible moves
    for (int i = 0; i < 4; ++i) {
        int newX = x + dx[i];
        int newY = y + dy[i];

        // Check if new position is within bounds and not blocked
        if (newX >= 0 && newX < N && newY >= 0 && newY < N && maze[newX][newY] == 1) {
            findPaths(maze, newX, newY, N, path + directions[i], result);
        }
    }

    // Unmark current cell before backtracking
    maze[x][y] = 1;
}

vector<string> ratInAMaze(vector<vector<int>>& maze, int N) {
    vector<string> result;
    // Check if source or destination is blocked
    if (maze[0][0] == 0 || maze[N - 1][N - 1] == 0) {
        result.push_back("-1");
        return result;
    }
    findPaths(maze, 0, 0, N, "", result);
    if (result.empty()) {
        result.push_back("-1");
    } else {
        // Sort the result in lexicographical order
        sort(result.begin(), result.end());
    }
    return result;
}

int main() {
    vector<vector<int>> maze1 = {{1, 0, 0, 0}, {1, 1, 0, 1}, {1, 1, 0, 0}, {0, 1, 1, 1}};
    int N1 = 4;

    vector<string> paths1 = ratInAMaze(maze1, N1);
    cout << "Example 1:\n";
    for (const string& path : paths1) {
        cout << path << " ";
    }
    cout << endl;

    vector<vector<int>> maze2 = {{1, 0}, {1, 0}};
    int N2 = 2;

    vector<string> paths2 = ratInAMaze(maze2, N2);
    cout << "Example 2:\n";
    for (const string& path : paths2) {
        cout << path << " ";
    }
    cout << endl;

    return 0;
}
*/

/*
//7.problem
#include <iostream>
#include <vector>
using namespace std;
bool isSafe(int v, vector<vector<int>>& graph, vector<int>& color, int c, int V) {
    for (int i = 0; i < V; ++i) {
        if (graph[v][i] && c == color[i])
            return false;
    }
    return true;
}

bool graphColoringUtil(vector<vector<int>>& graph, int m, vector<int>& color, int v, int V) {
    if (v == V)
        return true;

    for (int c = 1; c <= m; ++c) {
        if (isSafe(v, graph, color, c, V)) {
            color[v] = c;
            if (graphColoringUtil(graph, m, color, v + 1, V))
                return true;
            color[v] = 0;
        }
    }
    return false;
}

bool graphColoring(vector<vector<int>>& graph, int m, int V) {
    vector<int> color(V, 0);

    if (!graphColoringUtil(graph, m, color, 0, V))
        return false;

    return true;
}

int main() {
    int N1 = 4, M1 = 3, E1 = 5;
    vector<vector<int>> Edges1 = { {0,1},{1,2},{2,3},{3,0},{0,2} };
    vector<vector<int>> graph1(N1, vector<int>(N1, 0));
    for (auto edge : Edges1) {
        graph1[edge[0]][edge[1]] = 1;
        graph1[edge[1]][edge[0]] = 1;
    }
    cout << graphColoring(graph1, M1, N1) << endl;

    int N2 = 3, M2 = 2, E2 = 3;
    vector<vector<int>> Edges2 = { {0,1},{1,2},{0,2} };
    vector<vector<int>> graph2(N2, vector<int>(N2, 0));
    for (auto edge : Edges2) {
        graph2[edge[0]][edge[1]] = 1;
        graph2[edge[1]][edge[0]] = 1;
    }
    cout << graphColoring(graph2, M2, N2) << endl;

    return 0;
}
*/

/*
//8.problem
#include <iostream>
#include <vector>
#include <string>
using namespace std;

bool isValidPart(string s) {
    if (s.empty() || s.size() > 3 || (s.size() > 1 && s[0] == '0')) {
        return false;
    }
    int num = stoi(s);
    return num >= 0 && num <= 255;
}

void backtrack(string s, int start, int part, string current, vector<string>& result) {
    if (part == 4 && start == s.size()) {
        result.push_back(current);
        return;
    }
    if (part == 4 || start == s.size()) {
        return;
    }
    for (int len = 1; len <= 3 && start + len <= s.size(); len++) {
        string sub = s.substr(start, len);
        if (isValidPart(sub)) {
            backtrack(s, start + len, part + 1, current + (part == 0 ? "" : ".") + sub, result);
        }
    }
}

vector<string> genIp(string S) {
    vector<string> result;
    if (S.size() < 4 || S.size() > 12) {
        return { "-1" };
    }
    backtrack(S, 0, 0, "", result);
    return result;
}

int main() {
    string S = "1111";
    vector<string> result = genIp(S);
    for (string ip : result) {
        cout << ip << endl;
    }

    S = "55";
    result = genIp(S);
    for (string ip : result) {
        cout << ip << endl;
    }

    return 0;
}
*/

/*
//9.problem
#include <iostream>
#include <vector>
#include <unordered_set>
using namespace std;

bool isValid(int i, int j, int R, int C) {
    return (i >= 0 && i < R&& j >= 0 && j < C);
}

void dfs(vector<vector<char>>& board, vector<vector<bool>>& visited, int i, int j, string word, unordered_set<string>& dictionary, unordered_set<string>& result, int R, int C) {
    visited[i][j] = true;
    word += board[i][j];

    if (dictionary.find(word) != dictionary.end()) {
        result.insert(word);
    }

    
    int dx[] = { -1, -1, -1, 0, 0, 1, 1, 1 };
    int dy[] = { -1, 0, 1, -1, 1, -1, 0, 1 };

    for (int k = 0; k < 8; k++) {
        int ni = i + dx[k];
        int nj = j + dy[k];
        if (isValid(ni, nj, R, C) && !visited[ni][nj]) {
            dfs(board, visited, ni, nj, word, dictionary, result, R, C);
        }
    }

    visited[i][j] = false;
}

vector<string> wordBoggle(vector<vector<char>>& board, vector<string>& dictionary, int R, int C) {
    unordered_set<string> dict(dictionary.begin(), dictionary.end());
    unordered_set<string> result;
    vector<vector<bool>> visited(R, vector<bool>(C, false));

    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            string word = "";
            dfs(board, visited, i, j, word, dict, result, R, C);
        }
    }

    return vector<string>(result.begin(), result.end());
}

int main() {
    int R = 3, C = 3;
    vector<vector<char>> board = { {'C','A','P'},{'A','N','D'},{'T','I','E'} };
    vector<string> dictionary = { "CAT" };
    vector<string> result = wordBoggle(board, dictionary, R, C);
    for (auto word : result) {
        cout << word << " ";
    }
    cout << endl;

    R = 3; C = 3;
    board = { {'G','I','Z'},{'U','E','K'},{'Q','S','E'} };
    dictionary = { "GEEKS","FOR","QUIZ","GO" };
    result = wordBoggle(board, dictionary, R, C);
    for (auto word : result) {
        cout << word << " ";
    }
    cout << endl;

    return 0;
}
*/

/*
//10.problem
#include <iostream>
#include <vector>
using namespace std;

int kthElement(int arr1[], int arr2[], int N, int M, int K) {
    int i = 0, j = 0;
    int count = 0;
    int result = -1;

    while (i < N && j < M) {
        if (arr1[i] < arr2[j]) {
            count++;
            if (count == K) {
                result = arr1[i];
                break;
            }
            i++;
        }
        else {
            count++;
            if (count == K) {
                result = arr2[j];
                break;
            }
            j++;
        }
    }

   
    while (i < N && count < K) {
        count++;
        if (count == K) {
            result = arr1[i];
            break;
        }
        i++;
    }

    while (j < M && count < K) {
        count++;
        if (count == K) {
            result = arr2[j];
            break;
        }
        j++;
    }

    return result;
}

int main() {
    int arr1[] = { 2, 3, 6, 7, 9 };
    int arr2[] = { 1, 4, 8, 10 };
    int N = 5, M = 4, K = 5;
    cout << kthElement(arr1, arr2, N, M, K) << endl;

    int arr3[] = { 100, 112, 256, 349, 770 };
    int arr4[] = { 72, 86, 113, 119, 265, 445, 892 };
    N = 5, M = 7, K = 7;
    cout << kthElement(arr3, arr4, N, M, K) << endl;

    return 0;
}
*/

/*
//11.problem
#include <iostream>
#include <vector>
using namespace std;

bool isValid(int A[], int N, int M, int mid) {
    int students = 1;
    int sum = 0;
    for (int i = 0; i < N; i++) {
        sum += A[i];
        if (sum > mid) {
            students++;
            sum = A[i];
        }
        if (students > M) {
            return false;
        }
    }
    return true;
}

int findPages(int N, int A[], int M) {
    if (N < M) {
        return -1;
    }

    int total_pages = 0;
    int max_page = A[0];

    for (int i = 0; i < N; i++) {
        total_pages += A[i];
        if (A[i] > max_page) {
            max_page = A[i];
        }
    }

    int low = max_page;
    int high = total_pages;
    int result = INT_MAX;

    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (isValid(A, N, M, mid)) {
            result = min(result, mid);
            high = mid - 1;
        }
        else {
            low = mid + 1;
        }
    }

    return result == INT_MAX ? -1 : result;
}

int main() {
    int N1 = 4;
    int A1[] = { 12, 34, 67, 90 };
    int M1 = 2;
    cout << findPages(N1, A1, M1) << endl;

    int N2 = 3;
    int A2[] = { 15, 17, 20 };
    int M2 = 2;
    cout << findPages(N2, A2, M2) << endl;

    return 0;
}
*/

/*
//12.Problem
#include <iostream>
#include <vector>
using namespace std;

int numberSequence(int m, int n) {
    vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));

    // Base case: There's 1 way to form a sequence of length 1 with any number between 1 and m
    for (int i = 1; i <= m; ++i)
        dp[1][i] = 1;

    // Dynamic programming to fill the dp table
    for (int i = 2; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            for (int k = 1; k <= j / 2; ++k) {
                dp[i][j] += dp[i - 1][k];
            }
        }
    }

    // Sum up the last row to get the total number of sequences
    int total = 0;
    for (int i = 1; i <= m; ++i)
        total += dp[n][i];

    return total;
}

int main() {
    int m = 10, n = 4;
    cout << numberSequence(m, n) << endl; // Output: 4
    
    m = 5; n = 2;
    cout << numberSequence(m, n) << endl; // Output: 6
    
    return 0;
}
*/

