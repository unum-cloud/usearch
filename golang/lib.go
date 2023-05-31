package usearch

import (
	"errors"
	"fmt"
	"unsafe"
)

/*
#include <stdlib.h>
#include "usearch.h"
// avoid some verbosity by using void* index here
// proper typed pointer in the cpp file definitions
void* usearch_new_index(char* metric_str, int metric_len,
    char* accuracy_str, int accuracy_len,
    int dimensions, int capacity, int connectivity,
    int expansion_add, int expansion_search
);
void usearch_destroy(void* index);

const char* usearch_save(void* index, char* path);
const char* usearch_load(void* index, char* path);
const char* usearch_view(void* index, char* path);

//todo:: q:: is int<->size_t conversion portable?
int usearch_size(void* index);
int usearch_connectivity(void* index);
int usearch_dimensions(void* index);
int usearch_capacity(void* index);

const char* usearch_reserve(void* index, int capacity);
const char* usearch_add(void* index, int label, float* vec);
SearchResults usearch_search(void* index, float* query, int query_len, int limit);
#cgo CPPFLAGS: -I.
#cgo LDFLAGS: -lusearch
*/
import "C"

type DistMetric int8

const (
	L2_SQ DistMetric = iota
	IP
	COS
	Haversine
)

func (m DistMetric) String() string {
	switch m {
	case L2_SQ:
		return "l2sq"
	case IP:
		return "ip"
	case COS:
		return "cos"
	case Haversine:
		return "haversine"
	default:
		panic("unknown metric")
	}
}

type Accuracy int8

const (
	f32 Accuracy = iota
	f16
	f64
	i8q100
)

func (a Accuracy) String() string {
	switch a {
	case f16:
		return "f16"
	case f32:
		return "f32"
	case f64:
		return "f64"
	case i8q100:
		return "i8q100"
	default:
		panic("unknown accuracy")
	}
}

type IndexConfig struct {
	Accuracy        Accuracy
	Metric          DistMetric
	VecDimension    int
	InitCapacity    int
	Connectivity    int
	ExpansionAdd    int
	ExpansionSearch int
}

func DefaultConfig(dimension int) IndexConfig {
	c := IndexConfig{}
	c.InitCapacity = 32
	c.Connectivity = 16
	c.VecDimension = dimension
	c.ExpansionAdd = 128
	c.ExpansionSearch = 64
	return c
}

/*
Plan: read vectors in golang
pass them to C via a concurrent map
have the add() function copy over the vectors (maybe this copy is already done?? my job is easier)
have search()
pass around opeque native_ pointer between c and go
*/
type Index struct {
	opeque_handle *C.void
	config        IndexConfig
}

func NewIndex(conf IndexConfig) *Index {
	ind := &Index{config: conf}
	ind.init()
	return ind
}

func (ind *Index) init() {
	conf := ind.config
	metric_str := C.CString(conf.Metric.String())
	defer C.free(unsafe.Pointer(metric_str))
	metric_len := C.int(len(conf.Metric.String()))
	accuracy_str := C.CString(conf.Accuracy.String())
	defer C.free(unsafe.Pointer(accuracy_str))
	accuracy_len := C.int(len(conf.Accuracy.String()))
	dimensions := C.int(conf.VecDimension)
	capacity := C.int(conf.InitCapacity)
	connectivity := C.int(conf.Connectivity)
	expansion_add := C.int(conf.ExpansionAdd)
	expansion_search := C.int(conf.ExpansionSearch)
	ptr := C.usearch_new_index(metric_str, metric_len,
		accuracy_str, accuracy_len,
		dimensions, capacity, connectivity,
		expansion_add, expansion_search)

	ind.opeque_handle = (*C.void)(unsafe.Pointer(ptr))
}

func (ind *Index) Destroy() {
	if ind.opeque_handle == nil {
		panic("index not initialized")
	}
	C.usearch_destroy(unsafe.Pointer(ind.opeque_handle))
	ind.opeque_handle = nil
	ind.config = IndexConfig{}
}

func (ind *Index) fileOp(path string, op string) error {
	if ind.opeque_handle == nil {
		panic("index not initialized")
	}
	c_path := C.CString(path)
	defer C.free(unsafe.Pointer(c_path))
	var errStr *C.char
	switch op {
	case "load":
		errStr = C.usearch_load(unsafe.Pointer(ind.opeque_handle), c_path)
	case "save":
		errStr = C.usearch_save(unsafe.Pointer(ind.opeque_handle), c_path)
	case "view":
		errStr = C.usearch_view(unsafe.Pointer(ind.opeque_handle), c_path)
	default:
		panic("unknown file operation")
	}
	var err error
	if errStr != nil {
		err = errors.New(C.GoString(errStr))
		C.free(unsafe.Pointer(errStr))
	}
	return err
}

func (ind *Index) Save(path string) error {
	return ind.fileOp(path, "save")
}

func (ind *Index) Load(path string) error {
	return ind.fileOp(path, "load")
}

func (ind *Index) View(path string) error {
	return ind.fileOp(path, "view")
}

func (ind *Index) Len() int {
	return int(C.usearch_size(unsafe.Pointer(ind.opeque_handle)))
}

func (ind *Index) Connectivity() int {
	return int(C.usearch_connectivity(unsafe.Pointer(ind.opeque_handle)))
}

func (ind *Index) VecDimension() int {
	return int(C.usearch_dimensions(unsafe.Pointer(ind.opeque_handle)))
}

func (ind *Index) Capacity() int {
	return int(C.usearch_capacity(unsafe.Pointer(ind.opeque_handle)))
}

func (ind *Index) Reserve(capacity int) error {
	if ind.opeque_handle == nil {
		panic("index not initialized")
	}
	cap := ind.Capacity()
	if capacity < cap {
		return errors.New(fmt.Sprintf("cannot reserve to a value less than current capacity, current: %d, new: %d", cap, capacity))
	}
	errStr := C.usearch_reserve(unsafe.Pointer(ind.opeque_handle), (C.int)(capacity))
	var err error
	if errStr != nil {
		err = errors.New(C.GoString(errStr))
		C.free(unsafe.Pointer(errStr))
	}
	return err
}

func (ind *Index) Add(label int, vec []float32) error {
	if ind.opeque_handle == nil {
		panic("index not initialized")
	}
	errStr := C.usearch_add(unsafe.Pointer(ind.opeque_handle), (C.int)(label), (*C.float)(&vec[0]))
	var err error
	if errStr != nil {
		err = errors.New(C.GoString(errStr))
		C.free(unsafe.Pointer(errStr))
	}
	return err

}

// return must be int32 because int is 64bit in golang and 32bit in C
func (ind *Index) Search(query []float32, limit int) []int32 {
	if ind.opeque_handle == nil {
		panic("index not initialized")
	}
	if len(query) != ind.config.VecDimension {
		panic(fmt.Sprintf("query vector dimension mismatch. expected %d, got %d",
			ind.config.VecDimension, len(query)))
	}
	if limit <= 0 {
		panic("limit must be greater than 0")
	}
	res := C.usearch_search(unsafe.Pointer(ind.opeque_handle),
		(*C.float)(&query[0]), (C.int)(len(query)), (C.int)(limit))

	// my understanding is that search panics in truly exceptional cases,
	// none of which is every expected. so, not passing the error to the caller
	if res.Error != nil {
		defer C.free(unsafe.Pointer(res.Error))
		panic(C.GoString(res.Error))
	}
	var labs []int32
	//q:: who free's this memory? will golang do it?
	labs = unsafe.Slice((*int32)(unsafe.Pointer(res.Labels)), res.Len)
	return labs
}

func (ind *Index) Size() int {
	if ind.opeque_handle == nil {
		panic("index not initialized")
	}
	return int(C.usearch_size(unsafe.Pointer(ind.opeque_handle)))
}
