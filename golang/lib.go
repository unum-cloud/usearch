package usearch

import (
	"errors"
	"fmt"
	"unsafe"
)

/*
#include "../c/usearch.h"

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
	f8
)

func (a Accuracy) String() string {
	switch a {
	case f16:
		return "f16"
	case f32:
		return "f32"
	case f64:
		return "f64"
	case f8:
		return "f8"
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

type Index struct {
	opaque_handle *C.void
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
	ptr := C.usearch_new(metric_str, metric_len,
		accuracy_str, accuracy_len,
		dimensions, capacity, connectivity,
		expansion_add, expansion_search)

	ind.opaque_handle = (*C.void)(unsafe.Pointer(ptr))
}

func (ind *Index) Destroy() {
	if ind.opaque_handle == nil {
		panic("index not initialized")
	}
	C.usearch_free(unsafe.Pointer(ind.opaque_handle))
	ind.opaque_handle = nil
	ind.config = IndexConfig{}
}

func (ind *Index) fileOp(path string, op string) error {
	if ind.opaque_handle == nil {
		panic("index not initialized")
	}
	c_path := C.CString(path)
	defer C.free(unsafe.Pointer(c_path))
	var errStr *C.char
	switch op {
	case "load":
		errStr = C.usearch_load(unsafe.Pointer(ind.opaque_handle), c_path)
	case "save":
		errStr = C.usearch_save(unsafe.Pointer(ind.opaque_handle), c_path)
	case "view":
		errStr = C.usearch_view(unsafe.Pointer(ind.opaque_handle), c_path)
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
	return int(C.usearch_size(unsafe.Pointer(ind.opaque_handle)))
}

func (ind *Index) Connectivity() int {
	return int(C.usearch_connectivity(unsafe.Pointer(ind.opaque_handle)))
}

func (ind *Index) VecDimension() int {
	return int(C.usearch_dimensions(unsafe.Pointer(ind.opaque_handle)))
}

func (ind *Index) Capacity() int {
	return int(C.usearch_capacity(unsafe.Pointer(ind.opaque_handle)))
}

func (ind *Index) Reserve(capacity int) error {
	if ind.opaque_handle == nil {
		panic("index not initialized")
	}
	errStr := C.usearch_reserve(unsafe.Pointer(ind.opaque_handle), (C.int)(capacity))
	var err error
	if errStr != nil {
		err = errors.New(C.GoString(errStr))
		C.free(unsafe.Pointer(errStr))
	}
	return err
}

func (ind *Index) Add(label int, vec []float32) error {
	if ind.opaque_handle == nil {
		panic("index not initialized")
	}
	errStr := C.usearch_add(unsafe.Pointer(ind.opaque_handle), (C.int)(label), (*C.float)(&vec[0]))
	var err error
	if errStr != nil {
		err = errors.New(C.GoString(errStr))
		C.free(unsafe.Pointer(errStr))
	}
	return err

}

// return must be int32 because int is 64bit in golang and 32bit in C
func (ind *Index) Search(query []float32, limit int) []int32 {
	if ind.opaque_handle == nil {
		panic("index not initialized")
	}
	if len(query) != ind.config.VecDimension {
		panic(fmt.Sprintf("query vector dimension mismatch. expected %d, got %d",
			ind.config.VecDimension, len(query)))
	}
	if limit <= 0 {
		panic("limit must be greater than 0")
	}
	res := C.usearch_search(unsafe.Pointer(ind.opaque_handle),
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
	if ind.opaque_handle == nil {
		panic("index not initialized")
	}
	return int(C.usearch_size(unsafe.Pointer(ind.opaque_handle)))
}
