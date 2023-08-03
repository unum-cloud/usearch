package usearch

import (
	"runtime"
	"testing"
)

func Assure(err error) {
	if err != nil {
		panic(err)
	}
}
func TestUSearch(t *testing.T) {
	runtime.LockOSThread()

	// Initialize
	dim := uint(128)
	conf := DefaultConfig(dim)
	ind, err := NewIndex(conf)
	if err != nil {
		t.Fatalf("Couldn't construct the index: %s", err)
	}
	defer ind.Destroy()

	found_dims, err := ind.Dimensions()
	if err != nil {
		t.Fatalf("Couldn't retrieve dimensions: %s", err)
	}
	if found_dims != dim {
		t.Fatalf("Wrong number of dimensions")
	}

	found_len, err := ind.Len()
	if err != nil {
		t.Fatalf("Couldn't retrieve size: %s", err)
	}
	if found_len != 0 {
		t.Fatalf("Wrong size")
	}
	err = ind.Reserve(100)
	if err != nil {
		t.Fatalf("Couldn't reserve capacity: %s", err)
	}

	// Insert
	vec := make([]float32, dim)
	vec[0] = 40.0
	vec[1] = 2.0
	err = ind.Add(42, vec)
	if err != nil {
		t.Fatalf("Couldn't insert: %s", err)
	}
	found_len, err = ind.Len()
	if err != nil {
		t.Fatalf("Couldn't retrieve size: %s", err)
	}
	if found_len != 1 {
		t.Fatalf("Wrong size")
	}

	// Search
	keys, distances, err := ind.Search(vec, 10)
	if err != nil {
		t.Fatalf("Couldn't search: %s", err)
	}
	if keys[0] != 42 || distances[0] != 0.0 {
		t.Fatalf("Expected result 42")
	}
}
