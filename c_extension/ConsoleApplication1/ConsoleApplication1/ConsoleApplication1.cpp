// ConsoleApplication1.cpp : Defines the entry point for the console application.
//
#include "C:\Python27\include\Python.h"
#include "vcvarstall.bat"
//#include "stdafx.h"



/*
int main()
{
    return 0;
}
*/

static PyObject * module_function(PyObject *self, PyObject *args) {
	float a, b, c;
	if (!PyArg_ParseTuple(args, "ff", &a, &b))
		return NULL;
	c = a + b;
	return Py_BuildValue("f", c);
}

static PyMethodDef MyMethods[] = {
	{"add", module_function, METH_VARARGS, "Adds two numbers"},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initmymath(void) {
	(void)Py_InitModule3("mymath", MyMethods,
		"My doc of mymath");
}


//----------------
/*
static struct PyModuleDef mymathmodule = {
	PyModuleDef_HEAD_INIT,
	"mymath", "Mymath documentation",
	-1,
	MyMethods
};

PyMODINIT_FUNC
PyInit_mymath(void) {
	return PyModule_Create(&mymathmodule);
}
*/
