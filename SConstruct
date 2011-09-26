# -*- coding: utf-8 -*-

import os

env = Environment(
    CCFLAGS="-Wall -g -O0 -fopenmp -std=c++0x",
    LIBS=["gomp"],
    CPPPATH=[Dir(".")])
env.ParseConfig("pkg-config --cflags --libs opencv")
Export("env")
Default(env.SConscript("testings/SConscript"))

prefix = ARGUMENTS.get("prefix", "/usr/local")
prefix_include = os.path.join(prefix, "include")

Alias("install", Install(prefix_include, Dir("cvutil")))
