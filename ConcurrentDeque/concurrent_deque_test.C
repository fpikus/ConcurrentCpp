// Copyright (c) 2026 Fedor G. Pikus, fpikus@gmail.com
//  https://github.com/fpikus/LockFree
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
#include "concurrent_deque.h"
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <utility>
#include <memory>
#include <stdexcept>

TEST(ConcurrentAppendDequeTest, PushBackAndRead) {
    ConcurrentAppendDeque<int, 1024> deque;
    for (int i = 0; i < 10; ++i) {
        deque.push_back(i * 10);
    }
    EXPECT_EQ(10u, deque.size());
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(i * 10, deque[i]);
    }
}

TEST(ConcurrentAppendDequeTest, EmplaceBack) {
    ConcurrentAppendDeque<std::pair<int, double>, 1024> deque;
    deque.emplace_back(1, 2.5);
    EXPECT_EQ(1u, deque.size());
    EXPECT_EQ(1, deque[0].first);
    EXPECT_EQ(2.5, deque[0].second);
}

TEST(ConcurrentAppendDequeTest, ResizeGrow) {
    ConcurrentAppendDeque<int, 1024> deque;
    deque.resize(10);
    EXPECT_EQ(10u, deque.size());
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(0, deque[i]);
    }
    deque.push_back(100);
    EXPECT_EQ(11u, deque.size());
    EXPECT_EQ(100, deque[10]);
}

// Shrinking is deliberately a no-op (the container is append-only, so elements are
// never destroyed while the container is alive); size must remain unchanged.
TEST(ConcurrentAppendDequeTest, ResizeShrink) {
    ConcurrentAppendDeque<int, 1024> deque;
    deque.resize(10);
    deque.resize(5);
    EXPECT_EQ(10u, deque.size());
}

TEST(ConcurrentAppendDequeTest, Reserve) {
    ConcurrentAppendDeque<int, 1024> deque;
    deque.reserve(20);
    EXPECT_EQ(0u, deque.size());
    for (int i = 0; i < 20; ++i) {
        deque.push_back(i);
    }
    EXPECT_EQ(20u, deque.size());
    EXPECT_EQ(19, deque[19]);
}

TEST(ConcurrentAppendDequeTest, MultiThreadedWriters) {
    ConcurrentAppendDeque<int, 1024> deque;
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&deque]() {
            for (int j = 0; j < 1000; ++j) {
                deque.push_back(1);
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    EXPECT_EQ(4000u, deque.size());
}

// One writer appends 0..9999 while four wait-free readers race it. This is the
// canonical use of the reader protocol: read size() (acquire) first, then index
// only strictly below the value it returned. Because push_back publishes an
// element with a release store of size_, any index < the observed size names a
// fully constructed, visible element -- so `deque[idx]` here never reads a torn
// or unpublished value. `start` is a spin gate that lines the threads up so the
// reads and writes actually overlap.
TEST(ConcurrentAppendDequeTest, MultiThreadedReadersWriters) {
    ConcurrentAppendDeque<int, 1024> deque;
    std::atomic<bool> start{false};
    std::thread writer([&]() {
        while (!start) {}
        for (int i = 0; i < 10000; ++i) {
            deque.push_back(i);
        }
    });

    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&]() {
            while (!start) {}
            for (int j = 0; j < 100000; ++j) {
                // size() is the acquire load; idx = s - 1 is guaranteed < size(),
                // so the element it names is already published by the writer.
                size_t s = deque.size();
                if (s > 0) {
                    size_t idx = s - 1;
                    int val = deque[idx];
                    EXPECT_LT(val, 10000); // writer stores values 0..9999
                }
            }
        });
    }

    start = true;
    writer.join();
    for (auto& t : readers) {
        t.join();
    }
    EXPECT_EQ(10000u, deque.size());
    for (int i = 0; i < 10000; ++i) {
        EXPECT_EQ(i, deque[i]);
    }
}

// BlockSize 1 makes every push_back allocate a block and forces a directory
// reallocation on each capacity doubling, maximizing the window in which a reader
// holds a just-retired directory pointer. Verifies that retired directories remain
// dereferenceable (no use-after-free) for the container's lifetime.
TEST(ConcurrentAppendDequeTest, TSAN_StaleDirectory) {
    ConcurrentAppendDeque<int, 1> deque;
    std::atomic<bool> start{false};
    std::atomic<bool> done{false};

    std::thread writer([&]() {
        while (!start) {}
        for (int i = 0; i < 10000; ++i) {
            deque.push_back(i);
        }
        done = true;
    });

    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&]() {
            while (!start) {}
            while (!done) {
                size_t s = deque.size();
                if (s > 0) {
                    int val = deque[s / 2];
                    EXPECT_LT(val, 10000); // writer stores values 0..9999
                }
            }
        });
    }

    start = true;
    writer.join();
    for (auto& t : readers) {
        t.join();
    }
}

// A multi-word object whose fields satisfy the invariant b == a + 1 (and
// c == a + 2, d == a + 3) once fully constructed. It exists so the reader below
// can detect a *partially visible* construction: if the release/acquire pairing
// were wrong, a reader could observe the element's slot published (via size_)
// while some of its member stores had not yet propagated, and b == a + 1 would
// fail. A single scalar int could not expose that class of bug.
struct ComplexObject {
    int a, b, c, d;
    ComplexObject(int v = 0) : a(v), b(v+1), c(v+2), d(v+3) {}
};

// Verifies whole-object visibility, not just that size_ is synchronized: the
// writer emplaces ComplexObjects; each reader loads size() (acquire) and reads
// the last published object's a and b, asserting b == a + 1. The acquire on
// size() must make every member store of that object visible, or the invariant
// would tear. Passing under TSan/ASan is the real assertion here.
TEST(ConcurrentAppendDequeTest, TSAN_MemoryOrdering) {
    ConcurrentAppendDeque<ComplexObject, 64> deque;
    std::atomic<bool> start{false};
    std::atomic<bool> done{false};

    std::thread writer([&]() {
        while (!start) {}
        for (int i = 0; i < 10000; ++i) {
            deque.emplace_back(i);
        }
        done = true;
    });

    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&]() {
            while (!start) {}
            while (!done) {
                size_t s = deque.size();
                if (s > 0) {
                    size_t idx = s - 1;
                    const ComplexObject& obj = deque[idx];
                    int a = obj.a;
                    int b = obj.b;
                    EXPECT_EQ(b, a + 1);
                }
            }
        });
    }

    start = true;
    writer.join();
    for (auto& t : readers) {
        t.join();
    }
}

TEST(ConcurrentAppendDequeTest, BasicContainerMethods) {
    ConcurrentAppendDeque<int, 1024> deque;
    EXPECT_TRUE(deque.empty());
    EXPECT_EQ(0u, deque.capacity());

    deque.push_back(10);
    EXPECT_FALSE(deque.empty());
    EXPECT_EQ(10, deque.front());
    EXPECT_EQ(10, deque.back());
    EXPECT_EQ(10, deque.at(0));
    EXPECT_THROW(deque.at(1), std::out_of_range);

    deque.push_back(20);
    EXPECT_EQ(10, deque.front());
    EXPECT_EQ(20, deque.back());
    EXPECT_EQ(20, deque.at(1));
    EXPECT_THROW(deque.at(2), std::out_of_range);
    
    EXPECT_GE(deque.capacity(), 1024u);
}

TEST(ConcurrentAppendDequeTest, MoveSemantics) {
    ConcurrentAppendDeque<std::unique_ptr<int>, 1024> deque;
    deque.push_back(std::make_unique<int>(42));
    EXPECT_EQ(1u, deque.size());
    EXPECT_EQ(42, *deque[0]);
}

TEST(ConcurrentAppendDequeTest, IteratorBasic) {
    ConcurrentAppendDeque<int, 1024> deque;
    for (int i = 0; i < 5; ++i) {
        deque.push_back(i);
    }
    
    int expected = 0;
    for (int val : deque) {
        EXPECT_EQ(expected++, val);
    }
    EXPECT_EQ(5, expected);
}

TEST(ConcurrentAppendDequeTest, IteratorBoundary) {
    // BlockSize is 4 to force multiple block boundaries
    ConcurrentAppendDeque<int, 4> deque;
    for (int i = 0; i < 15; ++i) { // Crosses block boundaries at 4, 8, 12
        deque.push_back(i);
    }

    int expected = 0;
    for (auto it = deque.begin(); it != deque.end(); ++it) {
        EXPECT_EQ(expected++, *it);
    }
    EXPECT_EQ(15, expected);
}

TEST(ConcurrentAppendDequeTest, IteratorRandomAccess) {
    ConcurrentAppendDeque<int, 4> deque;
    for (int i = 0; i < 20; ++i) {
        deque.push_back(i);
    }

    auto it = deque.begin();
    EXPECT_EQ(0, *it);
    
    it += 5;
    EXPECT_EQ(5, *it);
    
    it -= 2;
    EXPECT_EQ(3, *it);
    
    auto it2 = deque.begin() + 10;
    EXPECT_EQ(10, *it2);
    
    EXPECT_EQ(7, it2 - it);
    EXPECT_TRUE(it < it2);
    EXPECT_TRUE(it2 > it);
    EXPECT_TRUE(it != it2);
}

// iterator must implicitly convert to const_iterator, e.g. to compare a mutable
// iterator against cbegin()/cend() or to pass it to a const_iterator parameter.
TEST(ConcurrentAppendDequeTest, ConstIteratorConversion) {
    ConcurrentAppendDeque<int, 4> deque;
    deque.push_back(1);
    deque.push_back(2);

    ConcurrentAppendDeque<int, 4>::const_iterator cit = deque.begin(); // implicit conversion
    EXPECT_EQ(1, *cit);
    ++cit;
    EXPECT_EQ(2, *cit);
    EXPECT_TRUE(deque.cbegin() + 1 == cit);
}

TEST(ConcurrentAppendDequeTest, IteratorEdgeCases) {
    ConcurrentAppendDeque<int, 4> empty_deque;
    EXPECT_TRUE(empty_deque.begin() == empty_deque.end());
    EXPECT_EQ(0, empty_deque.end() - empty_deque.begin());

    ConcurrentAppendDeque<int, 4> deque;
    deque.push_back(1);
    auto it = deque.begin();
    EXPECT_EQ(1, *it);
    it++;
    EXPECT_TRUE(it == deque.end());
}
