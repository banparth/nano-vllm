#pragma once
#include <Python.h>

// Emits a minimal `PyInit_<NAME>` so the compiled artifact is importable as a
// Python module. Importing it runs the translation unit's static initializers,
// which is what registers our TORCH_LIBRARY ops into the PyTorch dispatcher.
// This avoids pulling in pybind11 just to make the module loadable.
#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)
#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)
#define REGISTER_EXTENSION(NAME)                                              \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                    \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                \
                                        STRINGIFY(NAME), nullptr, 0,          \
                                        nullptr};                            \
    return PyModule_Create(&module);                                         \
  }
