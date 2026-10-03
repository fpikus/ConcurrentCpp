// Copyright (c) 2026 Fedor G. Pikus, fpikus@gmail.com
//  https://github.com/fpikus/ConcurrentCpp
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
#ifndef INCLUDED_HP_DRAIN_GTEST_H_
#define INCLUDED_HP_DRAIN_GTEST_H_

#include <gtest/gtest.h>

#include "hp_drain.h"

// Including this header in a GoogleTest binary runs force_early_scan()
// (hp_drain.h) once, before the first test, from a global test Environment.
//
// The registration is a namespace-scope variable initialized before main(), as
// GoogleTest documents for AddGlobalTestEnvironment(), so it needs no code in
// main() and works with libgtest_main. The variable is `inline`: however many
// TUs of one binary include this header, it is initialized, and the Environment
// registered, once. GoogleTest owns the Environment and deletes it at exit.
//
// SetUp() runs in the RUN_ALL_TESTS() process before any test; it runs again in
// the child of a "threadsafe"-style death test (which re-executes the binary), so
// every process that runs tests has done the early scan. The process is
// quiescent there (no test thread exists yet), as force_early_scan() requires.

// The global Environment whose SetUp() forces mm_hp's first scan.
class HpEarlyScanEnvironment : public ::testing::Environment {
public:
    void SetUp() override { force_early_scan(); }
}; // class HpEarlyScanEnvironment

// Registration handle; the pointer itself is not used.
inline ::testing::Environment* const hp_early_scan_environment =
    ::testing::AddGlobalTestEnvironment(new HpEarlyScanEnvironment);

#endif // INCLUDED_HP_DRAIN_GTEST_H_
