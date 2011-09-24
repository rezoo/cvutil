# -*- coding: utf-8 -*-

env = Environment(CCFLAGS="-Wall -O2 -fopenmp", LIBS=["gtest"], CPPPATH=["."])
env.ParseConfig("pkg-config --cflags --libs opencv")
env.Program("testings/all_test", Glob("testings/*.cpp"))
