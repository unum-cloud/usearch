package usearch

import (
	"errors"
	"unsafe"
)

/*
#cgo LDFLAGS: -L. -L/usr/local/lib -lusearch_c
#include "usearch.h"
#include <stdlib.h>
*/
import "C"

type Key = uint64

type Metric uint8

const (
	InnerProduct Metric = iota
	Cosine
	L2sq
	Haversine
	Divergence
	Pearson
	Hamming
	Tanimoto
	Sorensen
)

func (m Metric) String() string {
	switch m {
	case L2sq:
		return "l2sq"
	case InnerProduct:
		return "ip"
	case Cosine:
		return "cos"
	case Haversine:
		return "haversine"
	case Divergence:
		return "divergence"
	case Pearson:
		return "pearson"
	case Hamming:
		return "hamming"
	case Tanimoto:
		return "tanimoto"
	case Sorensen:
		return "sorensen"
	default:
		panic("Unknown metric")
	}
}

type Quantization uint8

const (
	F32 Quantization = iota
	F16
	F64
	I8
	B1
)

func (a Quantization) String() string {
	switch a {
	case F16:
		return "F16"
	case F32:
		return "F32"
	case F64:
		return "F64"
	case I8:
		return "I8"
	case B1:
		return "B1"
	default:
		panic("Unknown quantization")
	}
}

type IndexConfig struct {
	// The scalar kind used for quantization of vector data during indexing.
	Quantization Quantization
	// The metric kind used for distance calculation between vectors.
	Metric Metric
	// The number of dimensions in the vectors to be indexed.
	Dimensions uint
	// The @b optional connectivity parameter that limits connections-per-node in graph.
	Connectivity uint
	// The @b optional expansion factor used for index construction when adding vectors.
	ExpansionAdd uint
	// The @b optional expansion factor used for index construction during search operations.
	ExpansionSearch uint
	// The @b optional multi flag.
	Multi bool
}

func DefaultConfig(dimensions uint) IndexConfig {
	c := IndexConfig{}
	c.Dimensions = dimensions
	c.Metric = Cosine
	// Zeros will be replaced by the underlying C implementation
	c.Connectivity = 0
	c.ExpansionAdd = 0
	c.ExpansionSearch = 0
	c.Multi = false
	return c
}

type Index struct {
	opaque_handle *C.void
	config        IndexConfig
}

// Initializes a new instance of the index.
func NewIndex(conf IndexConfig) (index *Index, err error) {
	index = &Index{config: conf}

	conf = index.config
	dimensions := C.size_t(conf.Dimensions)
	connectivity := C.size_t(conf.Connectivity)
	expansion_add := C.size_t(conf.ExpansionAdd)
	expansion_search := C.size_t(conf.ExpansionSearch)
	multi := C.bool(conf.Multi)

	options := C.struct_usearch_init_options_t{}
	options.dimensions = dimensions
	options.connectivity = connectivity
	options.expansion_add = expansion_add
	options.expansion_search = expansion_search
	options.multi = multi

	// Map the metric kind to a C enum
	switch conf.Metric {
	case L2sq:
		options.metric_kind = C.usearch_metric_l2sq_k
	case InnerProduct:
		options.metric_kind = C.usearch_metric_ip_k
	case Cosine:
		options.metric_kind = C.usearch_metric_cos_k
	case Haversine:
		options.metric_kind = C.usearch_metric_haversine_k
	case Pearson:
		options.metric_kind = C.usearch_metric_pearson_k
	case Hamming:
		options.metric_kind = C.usearch_metric_hamming_k
	case Tanimoto:
		options.metric_kind = C.usearch_metric_tanimoto_k
	case Sorensen:
		options.metric_kind = C.usearch_metric_sorensen_k
	default:
		options.metric_kind = C.usearch_metric_unknown_k
	}

	// Map the quantization method
	switch conf.Quantization {
	case F16:
		options.quantization = C.usearch_scalar_f16_k
	case F32:
		options.quantization = C.usearch_scalar_f32_k
	case F64:
		options.quantization = C.usearch_scalar_f64_k
	case I8:
		options.quantization = C.usearch_scalar_i8_k
	case B1:
		options.quantization = C.usearch_scalar_b1_k
	default:
		options.quantization = C.usearch_scalar_unknown_k
	}

	var errorMessage *C.char
	ptr := C.usearch_init(&options, (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return nil, errors.New(C.GoString(errorMessage))
	}

	index.opaque_handle = (*C.void)(unsafe.Pointer(ptr))
	return index, nil
}

func (index *Index) Len() (len uint, err error) {
	var errorMessage *C.char
	len = uint(C.usearch_size((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		err = errors.New(C.GoString(errorMessage))
	}
	return len, err
}

func (index *Index) Connectivity() (con uint, err error) {
	var errorMessage *C.char
	con = uint(C.usearch_connectivity((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		err = errors.New(C.GoString(errorMessage))
	}
	return con, err
}

func (index *Index) Dimensions() (dim uint, err error) {
	var errorMessage *C.char
	dim = uint(C.usearch_dimensions((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		err = errors.New(C.GoString(errorMessage))
	}
	return dim, err
}

func (index *Index) Capacity() (cap uint, err error) {
	var errorMessage *C.char
	cap = uint(C.usearch_capacity((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		err = errors.New(C.GoString(errorMessage))
	}
	return cap, err
}

// Frees the resources associated with the index.
func (index *Index) Destroy() error {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	var errorMessage *C.char
	C.usearch_free((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	index.opaque_handle = nil
	index.config = IndexConfig{}
	return nil
}

// Reserves memory for a specified number of incoming vectors.
func (index *Index) Reserve(capacity uint) error {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	var errorMessage *C.char
	C.usearch_reserve((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (C.size_t)(capacity), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Adds a vector with a key to the index.
func (index *Index) Add(key Key, vec []float32) error {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	var errorMessage *C.char
	C.usearch_add((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (C.usearch_key_t)(key), unsafe.Pointer(&vec[0]), C.usearch_scalar_f32_k, (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Removes the vector associated with the given key from the index.
func (index *Index) Remove(key Key) error {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	var errorMessage *C.char
	C.usearch_remove((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (C.usearch_key_t)(key), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Checks if the index contains a vector with a specific key.
func (index *Index) Contains(key Key) (found bool, err error) {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	var errorMessage *C.char
	found = bool(C.usearch_contains((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (C.usearch_key_t)(key), (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		return found, errors.New(C.GoString(errorMessage))
	}
	return found, nil
}

func (index *Index) Get(key Key, count uint) (vectors []float32, err error) {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	vectors = make([]float32, index.config.Dimensions * count)
	var errorMessage *C.char
	found := uint(C.usearch_get((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (C.usearch_key_t)(key), (C.size_t)(count), unsafe.Pointer(&vectors[0]), C.usearch_scalar_f32_k, (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		return nil, errors.New(C.GoString(errorMessage))
	}
	if found == 0 {
		return nil, nil
	}
	return vectors, nil
}

// Performs k-Approximate Nearest Neighbors Search for closest vectors to query.
func (index *Index) Search(query []float32, limit uint) (keys []Key, distances []float32, err error) {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}
	if len(query) != int(index.config.Dimensions) {
		return nil, nil, errors.New("Number of dimensions doesn't match!")
	}

	keys = make([]Key, limit)
	distances = make([]float32, limit)
	var errorMessage *C.char
	count := uint(C.usearch_search((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), unsafe.Pointer(&query[0]), C.usearch_scalar_f32_k, (C.size_t)(limit), (*C.usearch_key_t)(&keys[0]), (*C.usearch_distance_t)(&distances[0]), (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		return nil, nil, errors.New(C.GoString(errorMessage))
	}

	keys = keys[:count]
	distances = distances[:count]
	return keys, distances, nil
}

// Saves the index to a file.
func (index *Index) Save(path string) error {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	c_path := C.CString(path)
	defer C.free(unsafe.Pointer(c_path))

	var errorMessage *C.char
	C.usearch_save((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), c_path, (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Loads the index from a file.
func (index *Index) Load(path string) error {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	c_path := C.CString(path)
	defer C.free(unsafe.Pointer(c_path))

	var errorMessage *C.char
	C.usearch_load((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), c_path, (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Creates a view of the index from a file without loading it into memory.
func (index *Index) View(path string) error {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	c_path := C.CString(path)
	defer C.free(unsafe.Pointer(c_path))

	var errorMessage *C.char
	C.usearch_view((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), c_path, (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}
